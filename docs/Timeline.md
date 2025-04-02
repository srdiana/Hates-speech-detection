### Project Timeline (6 Weeks)

---

#### **Week 1: Data Understanding & EDA**
- **Goals**: Load dataset, analyze class distribution, text length, missing values, and visualize key patterns.  
- **Progress**:  
  - ✅ **Done**: Basic EDA (class distribution, text length stats, missing values).  
  - ⏳ **Pending**: Deeper analysis (e.g., n-gram frequency, word clouds, correlation between text length and labels).  

---

#### **Week 2: Data Preprocessing & Feature Engineering**
- **Goals**: Tokenization, lemmatization, handling class imbalance (SMOTE/focal loss), and embedding preparation (GloVe).  
- **Progress**:  
  - ⚠️ **Not Started**: Class imbalance mitigation and embedding integration need implementation.  

---

#### **Week 3: Baseline Model (Logistic Regression)**  
- **Goals**: Train logistic regression with TF-IDF features; establish baseline metrics (precision, recall, F1).  
- **Progress**:  
  - ⚠️ **Not Started**: Requires preprocessing completion.  

---

#### **Week 4: BiLSTM Model Development**  
- **Goals**: Build BiLSTM architecture with attention layer; initialize training with GloVe embeddings.  
- **Progress**:  
  - ⚠️ **Not Started**: Dependent on Week 2’s preprocessing.  

---

#### **Week 5: Model Optimization & Comparison**  
- **Goals**: Fine-tune hyperparameters, evaluate BiLSTM against logistic regression, and interpret attention weights.  
- **Progress**:  
  - ⚠️ **Not Started**: Requires Week 3 and 4 outputs.  

---

#### **Week 6: Final Evaluation & Reporting**  
- **Goals**: Compile results, document insights, and prepare final report/presentation.  
- **Progress**:  
  - ⚠️ **Not Started**: Final phase.  

---

### **Key Risks & Mitigation**  
1. **Class Imbalance**: Use focal loss/SMOTE (planned in Week 2).  
2. **Contextual Nuances**: Ensure attention layer highlights discriminatory phrases (Week 4).  
3. **Computational Resources**: Optimize batch size/use cloud GPUs if needed.  

### **Next Immediate Steps**  
1. Address missing labels (1 NaN in `Label` column).  
2. Implement class imbalance handling (e.g., focal loss).  
3. Begin preprocessing (tokenization, GloVe embedding integration).  

This structured approach ensures alignment with the proposal’s goals while addressing real-world challenges in hate speech detection.