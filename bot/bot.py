import os
import logging
import requests
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
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

WEIGHTS_PATH     = os.getenv("MODEL_WEIGHTS_PATH", "balanced_nn_weights.pth")
LABELS_PATH      = os.getenv("LABELS_PATH", "label_classes.npy")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    str(Path("/Users/sozlaa/Downloads/embedding_model").resolve())
)

if not TELEGRAM_TOKEN or not MOVIE_API_KEY:
    logger.error("TELEGRAM_TOKEN and MOVIE_API_KEY must be set")
    exit(1)

HEADERS = {"X-API-KEY": MOVIE_API_KEY, "Accept": "application/json"}

# ——— Device setup —————————————————————————————————————————————————————————————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— 1) Load embedding model and determine dimension ——————————————————————————————————————————
logger.info(f"Loading embedding model from {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
INPUT_DIM = embedder.get_sentence_embedding_dimension()
logger.info(f"Detected embedding dimension: {INPUT_DIM}")

# ——— 2) Load class labels —————————————————————————————————————————————————————————————————
classes = list(np.load(LABELS_PATH, allow_pickle=True))
NUM_CLASSES = len(classes)
logger.info(f"Loaded {NUM_CLASSES} classes: {classes}")

# ——— 3) Define and load the classification model ——————————————————————————————————————————————
class BalancedNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)

model = BalancedNN(INPUT_DIM, NUM_CLASSES).to(device)
logger.info(f"Loading model weights from {WEIGHTS_PATH}")
state = torch.load(WEIGHTS_PATH, map_location=device)
# Adjust first layer if dimension mismatch
w0 = state["model.0.weight"]
old_dim = w0.size(1)
if old_dim != INPUT_DIM:
    logger.warning(f"Adjusting first layer: old_dim={old_dim}, new_dim={INPUT_DIM}")
    new_w0 = torch.zeros((w0.size(0), INPUT_DIM), device=w0.device)
    new_w0[:, :old_dim] = w0
    nn.init.xavier_uniform_(new_w0[:, old_dim:])
    state["model.0.weight"] = new_w0
model.load_state_dict(state)
model.eval()

# ——— 4) Classification helper —————————————————————————————————————————————————————————————————
def classify_reviews(reviews):
    counts = Counter()
    if not reviews:
        return counts
    embeddings = embedder.encode(reviews, convert_to_tensor=True)
    with torch.no_grad():
        logits = model(embeddings)
    preds = logits.argmax(dim=1).cpu().tolist()
    for p in preds:
        counts[classes[p]] += 1
    return counts

# ——— API helper functions —————————————————————————————————————————————————————————————————————
def search_movies(query, limit=5, page=1):
    url = f"{API_BASE_URL}/v1.4/movie/search"
    resp = requests.get(
        url, headers=HEADERS, params={"query": query, "limit": limit, "page": page}, timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("docs", [])

def get_reviews(movie_id, limit=20, page=1):
    url = f"{API_BASE_URL}/v1.4/review"
    resp = requests.get(
        url,
        headers=HEADERS,
        params={"movieId": movie_id, "limit": limit, "page": page},
        timeout=10
    )
    resp.raise_for_status()
    docs = resp.json().get("docs", [])
    texts = []
    for d in docs:
        if isinstance(d, dict):
            for key in ("reviewText", "review", "text", "comment"):
                v = d.get(key)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
                    break
        elif isinstance(d, str):
            texts.append(d.strip())
    return texts

# ——— Bot handlers —————————————————————————————————————————————————————————————————————————————
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! Please enter a movie title:")
    return SEARCH

def search_handler(update: Update, context: CallbackContext):
    q = update.message.text.strip()
    try:
        docs = search_movies(q, limit=10)
    except Exception:
        update.message.reply_text("Search error. Please try again later.")
        return ConversationHandler.END

    if not docs:
        update.message.reply_text("No movies found. Try a different title.")
        return SEARCH

    movies = []
    for m in docs:
        mid = m.get("id") or m.get("kinopoiskId")
        name = m.get("name") or m.get("nameRu") or m.get("nameEn") or str(mid)
        movies.append({"id": mid, "title": name})

    if len(movies) == 1:
        mv = movies[0]
        revs = get_reviews(mv["id"])
        counts = classify_reviews(revs)
        total = len(revs)
        if total == 0:
            update.message.reply_text(f"<b>{mv['title']}</b>\nNo reviews available.", parse_mode="HTML")
        else:
            lines = []
            for label, cnt in counts.items():
                pct = cnt / total * 100
                lines.append(f"{label}: {cnt} ({pct:.1f}%)")
            dist = "\n".join(lines)
            top_label, top_cnt = counts.most_common(1)[0]
            top_pct = top_cnt / total * 100
            update.message.reply_text(
                f"<b>{mv['title']}</b>\n"
                f"Total reviews: {total}\n\n"
                f"Sentiment distribution:\n{dist}\n\n"
                f"Most frequent: {top_label} ({top_cnt}, {top_pct:.1f}% of reviews)",
                parse_mode="HTML"
            )
        return ConversationHandler.END

    buttons = [
        [InlineKeyboardButton(m["title"], callback_data=str(m["id"]))]
        for m in movies[:5]
    ]
    update.message.reply_text(
        "Multiple movies found, please select one:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )
    return SELECT

def select_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    mid = int(query.data)
    query.answer()

    revs = get_reviews(mid)
    counts = classify_reviews(revs)
    total = len(revs)
    if total == 0:
        text = "No reviews available."
    else:
        lines = []
        for label, cnt in counts.items():
            pct = cnt / total * 100
            lines.append(f"{label}: {cnt} ({pct:.1f}%)")
        dist = "\n".join(lines)
        top_label, top_cnt = counts.most_common(1)[0]
        top_pct = top_cnt / total * 100
        text = (
            f"Total reviews: {total}\n\n"
            f"Sentiment distribution:\n{dist}\n\n"
            f"Most frequent: {top_label} ({top_cnt}, {top_pct:.1f}% of reviews)"
        )

    query.edit_message_text(text, parse_mode="HTML")
    return ConversationHandler.END

def cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Cancelled.")
    return ConversationHandler.END

# ——— Main ———————————————————————————————————————————————————————————————————————————————
def main():
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SEARCH: [MessageHandler(Filters.text & ~Filters.command, search_handler)],
            SELECT: [CallbackQueryHandler(select_handler)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True
    )
    dp.add_handler(conv)

    updater.start_polling()
    logger.info("Bot started.")
    updater.idle()

if __name__ == "__main__":
    main()






