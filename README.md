# PySphere Movie Review Sentiment Analysis

**COMP5423 Natural Language Processing - Lab1 Homework**

**Student**: Xinyu Zhu
**Student ID**: 25118165g  
**Competition**: PySphere Movie Review Sentiment Challenge  
**Kaggle Profile**: [jujubacon](https://www.kaggle.com/jujubacon)

A comprehensive sentiment classification project achieving **perfect accuracy (1.000)** and **ranking 7th** on the Kaggle leaderboard through multiple machine learning approaches.

---

## 📊 Project Overview

This project tackles binary sentiment classification on 2,000 movie reviews using multiple approaches from traditional ML to state-of-the-art deep learning:

| Model | Approach | Kaggle Score | Rank | GPU Required | Training Time |
|-------|----------|--------------|------|--------------|---------------|
| **BoW + Logistic Regression** | Traditional ML | 0.813 | - | No | ~25s |
| **TF-IDF + Logistic Regression** | Traditional ML | 0.813 | - | No | ~25s |
| **DeBERTa-v3** ⭐ | Pre-trained Transformer | **1.000** | **7th** | Yes (P100) | ~8.5min |

---

## 📁 Repository Structure

```
25118165g_Lab1/
├── README.md                                    # Project documentation
├── NLP_ASS.pdf                                  # Technical report
├── 25118165g_predictions.csv                    # Final submission (DeBERTa-v3, Score: 1.000)
│
├── Logistic Regression with Bow/
│   ├── BoW + Logistic Regression.ipynb         # Code implementation
│   └── ZHU_Xinyu_25118165g_predictions.csv     # Submission (Score: 0.813)
│
├── Logistic Regression with TF-IDF/
│   ├── TF-IDF+logistic regression.ipynb        # Code implementation
│   └── ZHU_Xinyu_25118165g_predictions.csv     # Submission (Score: 0.813)
│
└── DeBERTa_v3/
    ├── deberta_v3.ipynb                        # Code implementation
    └── deberta_submission.csv                  # Submission (Score: 1.000, Rank: 7th)
```

---

## 🎯 Task Description

- **Dataset**: 2,000 labeled movie reviews (balanced: 50% positive, 50% negative)
- **Goal**: Build ML models to predict sentiment (0=negative, 1=positive)
- **Evaluation Metric**: Accuracy
- **Competition**: [Kaggle PySphere Movie Review Sentiment Challenge](https://www.kaggle.com/competitions/py-sphere-movie-review-sentiment-challenge)
- **Final Achievement**: 100% accuracy, ranked 7th on public leaderboard

---

## 🚀 How to Run

All models can be run directly on **Kaggle Notebooks** with no local setup required!

### Model 1: Bag of Words (BoW) + Logistic Regression

**GitHub Repository**: https://github.com/Zhuxinyu0809/NLP_Lab.git

**Local Setup**:
1. Clone or download the repository
2. Install Python 3.8+
3. Install Jupyter Notebook: `pip install jupyter`
4. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
5. Navigate to `Logistic Regression with Bow/`
6. Run: `jupyter notebook "BoW + Logistic Regression.ipynb"`
7. Execute all cells (Ctrl+Enter)

**Kaggle Notebook**: https://www.kaggle.com/code/jujubacon/bow-logistic-regression

**On Kaggle**:
1. Open the link → Click **"Copy & Edit"**
3. Add competition data: **"Add Input"** → **"py-sphere-movie-review-sentiment-challenge"**
4. Click **"Run All"**
4. Download predictions from Output

**Technical Details**:
- **Preprocessing**: Tokenization, lowercasing, stop word removal
- **Vectorization**: CountVectorizer (bag-of-words)
- **Model**: Logistic Regression

**Requirements**: CPU only  
**Training Time**: ~25 seconds  
**Expected Score**: 0.813

---

### Model 2: TF-IDF + Logistic Regression

**GitHub Repository**: https://github.com/Zhuxinyu0809/NLP_Lab.git

**Local Setup (Optional)**:
1. Same as Model 1
2. Navigate to `Logistic Regression with TF-IDF/`
3. Run: `jupyter notebook "TF-IDF+logistic regression.ipynb"`

**Kaggle Notebook**: https://www.kaggle.com/code/jujubacon/tf-idf-logistic-regression

**On Kaggle**:
1. Open the link → Click **"Copy & Edit"**
3. Add competition data: **"Add Input"** → **"py-sphere-movie-review-sentiment-challenge"**
4. Click **"Run All"**
4. Download predictions from output

**Technical Details**:
- **Preprocessing**: Advanced text normalization
- **Vectorization**: TfidfVectorizer (term frequency-inverse document frequency)
- **Model**: Logistic Regression

**Requirements**: CPU only  
**Training Time**: ~25 seconds  
**Expected Score**: 0.813

---

### Model 3: DeBERTa-v3 (Best Model) ⭐

**Kaggle Notebook**: https://www.kaggle.com/code/jujubacon/deberta-v3

**Steps**:
1. Open the link → Click **"Copy & Edit"**
2. **⚠️ IMPORTANT**: Enable GPU
   - Right panel → **Settings** → **Accelerator** → Select **"GPU P100"** or **"GPU T4"**
3. Add competition data: **"Add Input"** → **"py-sphere-movie-review-sentiment-challenge"**
4. Click **"Run All"**
5. Wait ~8-10 minutes for training to complete
6. Download `deberta_submission.csv` from Output section

**Technical Details**:
- **Model**: microsoft/deberta-v3-base (86M parameters)
- **Architecture**: Decoding-enhanced BERT with disentangled attention
- **Why it works**: 
  - Pre-trained on massive text corpora
  - Understands context, negations, and subtle sentiment
  - Disentangled attention captures word relationships better

**Requirements**: GPU required (P100 recommended, T4 also works)  
**Training Time**: ~8.5 minutes  
**Expected Score**: 1.000

---

## 📈 Performance Analysis

### Comparative Results

| Metric | BoW | TF-IDF | DeBERTa-v3 |
|--------|-----|--------|---------|
| **Accuracy** | 0.813 | 0.813 | **1.000** |
| **Training Time** | 25s | 25s | 8.5min |
| **Model Size** | <1MB | <1MB | ~350MB |
| **Inference Speed** | Very Fast | Very Fast | Moderate |

### Key Findings

1. **Traditional ML Performance (BoW & TF-IDF)**:
   - Both achieved 81.3% accuracy
   - Fast training and inference
   - Good for resource-constrained environments
   - Limited by inability to understand context and word order

2. **Deep Learning Dominance (DeBERTa-v3)**:
   - Perfect 100% accuracy demonstrates transformer superiority
   - 18.7% improvement over traditional methods
   - Worth the computational cost for production systems
   - Captures semantic nuances traditional methods miss

---

## 🛠️ Technical Setup

### Environment

All code runs on **Kaggle Notebooks** with pre-installed dependencies:
- Python 3.10+
- transformers 4.30+
- scikit-learn 1.3+
- xgboost 2.0+
- lightgbm 4.0+
- torch 2.0+ (with CUDA support)
- pandas, numpy, matplotlib

**No manual installation required** on Kaggle - just click "Run All"!


## 📤 Kaggle Submission Files

1. **Final_predictions.csv** - Final submission (DeBERTa-v3, Score: 1.000)
2. **ZHU_Xinyu_25118165g_predictions.csv** - BoW/TF-IDF submissions (Score: 0.813)
3. **deberta_submission.csv** - DeBERTa-v3 (Score: 1.000, Rank: 7th)
   
---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/Zhuxinyu0809/NLP_Lab.git |
| **Kaggle Profile** | https://www.kaggle.com/code/jujubacon |
| **DeBERTa-v3 Notebook** | https://www.kaggle.com/code/jujubacon/deberta-v3 |
| **TF-IDF +Logistic Regression Notebook** | https://www.kaggle.com/code/jujubacon/tf-idf-logistic-regression |
| **Bow +Logistic Regression Notebook** | https://www.kaggle.com/code/jujubacon/bow-logistic-regression |
| **Competition Page** | https://www.kaggle.com/competitions/py-sphere-movie-review-sentiment-challenge |

---

## 📧 Contact

**Student**: Xinyu Zhu  
**Student ID**: 25118165g  
**Course**: COMP5423 Natural Language Processing  
**Kaggle**: [jujubacon](https://www.kaggle.com/code/jujubacon)  
**GitHub**: [Zhuxinyu0809](https://github.com/Zhuxinyu0809)
