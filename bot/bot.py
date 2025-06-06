import os
import logging
import requests
import numpy as np
from pathlib import Path
from collections import Counter
import random
from lime.lime_text import LimeTextExplainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    Filters,
    CallbackContext,
)

# ——— Настройка логов —————————————————————————————————————————————————————————————————
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ——— Состояния —————————————————————————————————————————————————————————————————
SEARCH, SELECT = range(2)

# ——— Параметры окружения —————————————————————————————————————————————————————————————————————
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
API_BASE_URL        = os.getenv("MOVIE_API_BASE_URL", "https://api.kinopoisk.dev")
MOVIE_API_KEY       = os.getenv("MOVIE_API_KEY", "")

WEIGHTS_PATH     = os.getenv("MODEL_WEIGHTS_PATH", "/Users/sozlaa/Hates-speech-detection/bot/ML/best_model (1).pth")
LABELS_PATH      = os.getenv("LABELS_PATH", "/Users/sozlaa/Hates-speech-detection/bot/ML/label_classes (3).npy")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    str(Path("/Users/sozlaa/Hates-speech-detection/bot/ML/embedding_model_312").resolve())
) 

if not TELEGRAM_TOKEN or not MOVIE_API_KEY:
    logger.error("Set TELEGRAM_TOKEN and MOVIE_API_KEY")
    exit(1)

HEADERS = {"X-API-KEY": MOVIE_API_KEY, "Accept": "application/json"}
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HELP_TEXT = (
    "Привет! Я — бот для классификации отзывов о фильмах.\n\n"
    "Доступные команды:\n"
    "/start   — начать работу и выбрать фильм\n"
    "/cancel  — отменить текущий запрос\n"
    "/explain — получить ключевые слова LIME по последнему отзыву\n"
    "/help    — показать это сообщение ещё раз\n"
)

# ——— 1) Эмбедер ———————————————————————————————————————————————————————————————
logger.info(f"Loading embedder from {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
EMB_DIM = embedder.get_sentence_embedding_dimension()
logger.info(f"Embedder dimension: {EMB_DIM}")  # = 384

# ——— 2) Метки ———————————————————————————————————————————————————————————————
classes = list(np.load(LABELS_PATH, allow_pickle=True))
NUM_CLASSES = len(classes)
logger.info(f"Classes: {classes}")

# ——— 3) BiLSTM_CNN с проекцией ————————————————————————————————————————————————————
class BiLSTM_CNN(nn.Module):
    def __init__(self,
                 embedding_dim,  # 384
                 proj_dim=312,   # то, что чекпоинт видел
                 hidden_dim=256, # то, что чекпоинт видел
                 num_classes=3,
                 n_filters=100,
                 filter_sizes=(1,3,5),
                 dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, proj_dim)
        self.lstm = nn.LSTM(
            proj_dim, hidden_dim,
            bidirectional=True, batch_first=True, dropout=0.3
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(2*hidden_dim, n_filters, fs, padding=(fs-1)//2)
            for fs in filter_sizes
        ])
        self.bn   = nn.BatchNorm1d(len(filter_sizes)*n_filters)
        self.fc   = nn.Linear(len(filter_sizes)*n_filters, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim()==2:
            x = x.unsqueeze(1)
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out    = lstm_out.permute(0,2,1)
        pooled = []
        for conv in self.convs:
            y = F.relu(conv(lstm_out))
            y = F.max_pool1d(y, y.size(2))
            pooled.append(y)
        cat = torch.cat(pooled, dim=1).squeeze(2)
        cat = self.bn(cat)
        return self.fc(self.drop(cat))

model = BiLSTM_CNN(embedding_dim=EMB_DIM).to(device)

# ——— 4) Загрузка весов с фильтрацией —————————————————————————————————————————————
logger.info(f"Loading weights from {WEIGHTS_PATH}")
ckpt       = torch.load(WEIGHTS_PATH, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt)
# фильтруем только совпадающие по имени и размеру тензоры
own_state = model.state_dict()
filtered  = {
    k: v for k, v in state_dict.items()
    if k in own_state and v.shape == own_state[k].shape
}
model.load_state_dict(filtered, strict=False)
model.eval()

explainer = LimeTextExplainer(class_names=classes)

def predict_proba(texts):
    """
    texts: list[str]
    возвращает np.array shape (n_samples, n_classes) с вероятностями
    """
    embs = embedder.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=False)
    with torch.no_grad():
        logits = model(embs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
    return probs

# ——— 5) classify_reviews и остальной код бота —————————————————————————————————————
def classify_reviews(reviews):
    counts = Counter()
    if not reviews:
        return counts
    embs = embedder.encode(
        reviews, convert_to_tensor=True,
        device=device, show_progress_bar=False
    )
    with torch.no_grad():
        logits = model(embs)
        preds  = logits.argmax(dim=1).cpu().tolist()
    for p in preds:
        counts[classes[p]] += 1
    return counts
# ——— 6) API helpers —————————————————————————————————————————————————————————————
def search_movies(query, limit=10, page=1):
    resp = requests.get(
        f"{API_BASE_URL}/v1.4/movie/search",
        headers=HEADERS,
        params={"query": query, "limit": limit, "page": page},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("docs", [])

def get_reviews(movie_id, limit=50):
    all_texts, page = [], 1
    while True:
        resp = requests.get(
            f"{API_BASE_URL}/v1.4/review",
            headers=HEADERS,
            params={"movieId": movie_id, "limit": limit, "page": page},
            timeout=10
        )
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        if not docs:
            break
        for d in docs:
            if isinstance(d, dict):
                for key in ("reviewText","review","text","comment"):
                    v = d.get(key)
                    if isinstance(v, str) and v.strip():
                        all_texts.append(v.strip())
                        break
            elif isinstance(d, str):
                all_texts.append(d.strip())
        page += 1

    return all_texts if len(all_texts) <= limit else random.sample(all_texts, limit)

# ——— 7) Форматирование и хендлеры —————————————————————————————————————————————————————
def _format_review_caption(details, counts, total):
    lines = [f"<b>{details['title']}</b>"]

    # Описание
    if desc := details.get("description"):
        if len(desc) > 300:
            desc = desc[:300].rsplit(" ",1)[0] + "…"
        lines.append(f"<i>{desc}</i>\n")

    # Легенда
    lines.append("0 — negative, 1 — neutral, 2 — positive\n")

    # Общая статистика
    lines.append(f"<b>Total reviews:</b> {total}")
    lines.append(f"<b>Distribution:</b>")

    # Для каждого числового кода берём строковое имя classes[lbl]
    for lbl in (2, 1, 0):
        name = classes[lbl]           # например classes[2] == "positive"
        cnt  = counts.get(name, 0)    # теперь ищем counts["positive"]
        pct  = cnt / total * 100 if total else 0.0
        lines.append(f"  {lbl} — {name}: {cnt} ({pct:.1f}%)")

    return "\n".join(lines)


def _show_movie_reviews(target, context: CallbackContext, movie_id: int):
    details = context.user_data["movies_map"][movie_id]
    revs    = get_reviews(movie_id, limit=50)
    context.user_data["last_reviews"] = revs
    counts  = classify_reviews(revs)
    cap     = _format_review_caption(details, counts, len(revs))
    if details.get("poster"):
        target.reply_photo(photo=details["poster"], caption=cap, parse_mode="HTML")
    else:
        target.reply_text(cap, parse_mode="HTML")
    return ConversationHandler.END

def start(update: Update, context: CallbackContext):
    # Сначала показываем помощь
    update.message.reply_text(HELP_TEXT, parse_mode="HTML")
    # Затем приглашаем ввести название
    update.message.reply_text("Введите название фильма для поиска:")
    return SEARCH


def help_handler(update: Update, context: CallbackContext):
    update.message.reply_text(HELP_TEXT, parse_mode="HTML")

def search_handler(update: Update, context: CallbackContext):
    q = update.message.text.strip()
    try:
        docs = search_movies(q, limit=10)
    except:
        update.message.reply_text("Ошибка поиска. Попробуйте позже.")
        return ConversationHandler.END

    if not docs:
        update.message.reply_text("Ничего не найдено.")
        return SEARCH

    kb = []
    context.user_data["movies_map"] = {}
    for m in docs[:5]:
        mid   = m.get("id") or m.get("kinopoiskId")
        title = m.get("name") or m.get("nameRu") or m.get("nameEn") or str(mid)
        desc  = m.get("description","").strip()
        poster= m.get("poster",{}).get("url")
        context.user_data["movies_map"][mid] = {
            "title": title, "description": desc, "poster": poster
        }
        kb.append([InlineKeyboardButton(title, callback_data=str(mid))])

    if len(kb) == 1:
        return _show_movie_reviews(update.message, context, mid)

    update.message.reply_text("Выберите фильм:", reply_markup=InlineKeyboardMarkup(kb))
    return SELECT

def select_handler(update: Update, context: CallbackContext):
    q = update.callback_query; q.answer()
    return _show_movie_reviews(q.message, context, int(q.data))

def cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Отменено.")
    return ConversationHandler.END

def explain_handler(update: Update, context: CallbackContext):
    # возьмём сохранённые в user_data отзывы последней сессии
    revs = context.user_data.get("last_reviews", [])
    if not revs:
        return update.message.reply_text("Сначала выберите фильм и получите отзывы.")

    text = revs[0]  # или объединить несколько отзывов
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=5,
        num_samples=100
    )
    kws = [w for w,_ in exp.as_list()]
    return update.message.reply_text(
        "<b>Key words:</b> " + ", ".join(kws),
        parse_mode="HTML"
    )

def main():
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher

    # 1) Общие команды
    dp.add_handler(CommandHandler("help", help_handler))
    dp.add_handler(CommandHandler("cancel", cancel))
    dp.add_handler(CommandHandler("explain", explain_handler))

    # 2) Conversation для /start → поиск → выбор фильма
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SEARCH: [MessageHandler(Filters.text & ~Filters.command, search_handler)],
            SELECT: [CallbackQueryHandler(select_handler)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )
    dp.add_handler(conv)

    updater.start_polling()
    logger.info("Bot started.")
    updater.idle()

if __name__ == "__main__":
    main()
