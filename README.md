
# COMP5423 NLP Lab 1 - README

**Student ID**: 25118165g  
**Name**: Xinyu Zhu  
**Competition**: PySphere Movie Review Sentiment Challenge

---

##  Submission Structure

```
25118165g_Lab1/
├── README.md                              # This file
├── NLP_ASS.pdf                           # Technical report
├── 25118165g_predictions.csv            # Final submission (DeBERTa-v3, Score: 1.000)
│
├── Logistic Regression with Bow
│   ├── BoW + Logistic Regression.ipynb       # Code notebook
│   └── ZHU_Xinyu_25118165g_predictions.csv    # Submission (Score: 0.813)
│
├── Logistic Regression with TF-IDF
│   ├── TF-IDF+logistic regression.ipynb                 # Code notebook
│   └── ZHU_Xinyu_25118165g_predictions.csv       # Submission (Score: 0.813)
│
└── DeBERTa_v3
    ├── deberta_v3.ipynb                 # Code notebook
    └── deberta_submission.csv           # Submission (Score: 1.000, Rank 7th)
```

---

## 🚀 How to Run (on Kaggle)

### Model 1: Logistic Regression with TF-IDF

**Github**: https://github.com/Zhuxinyu0809/NLP_Lab.git

**Steps**:
1. Open the link → Click **"code"** → Click **"Download ZIP"**
2. Install python 3
3. Install **"Jupyter Notebook"** or **"Jupyter Lab"** (to open and run .ipynb files). You can install it via pip
```pip install jupyter```
5. In right panel: **"Add Data"** → Search **"py-sphere-movie-review-sentiment-challenge"** → Click **"Add"**
6. Click **"Run All"** (or press Ctrl+Enter for each cell)
7. Download `submission.csv` from Output section

**Requirements**: No GPU needed  
**Training Time**: ~25 seconds  
**Expected Score**: 0.813

---

### Model 2: XGBoost/LightGBM Ensemble

**Kaggle Notebook**: https://www.kaggle.com/code/moonquakemiao/notebook52d36206ec

**Steps**:
1. Open the link → Click **"Copy & Edit"**
2. Add competition data (same as Model 1)
3. Click **"Run All"**
4. Download `ensemble_submission.csv`

**Requirements**: No GPU needed  
**Training Time**: ~35 seconds  
**Expected Score**: 0.736

---

### Model 3: DeBERTa-v3 (Best Model - 7th Place)

**Kaggle Notebook**: https://www.kaggle.com/code/jujubacon/deberta-v3

**Steps**:
1. Open the link → Click **"Copy & Edit"**
2. **Important**: In right panel → **Settings** → **Accelerator** → Select **"GPU P100"**
3. Add competition data: **"Add Input"** → **"py-sphere-movie-review-sentiment-challenge"**
4. Click **"Run All"**
5. Wait ~8-10 minutes for training
6. Download `deberta_submission.csv`

**Requirements**: GPU required (T4 or P100)  
**Training Time**: ~8 minutes  
**Expected Score**: 1.000

---

## 📊 Results Summary

| Model | Kaggle Score | Rank | GPU Required | Training Time |
|-------|--------------|------|--------------|---------------|
| Logistic Regression + TF-IDF | 0.813 | - | No | ~25s |
| XGBoost/LightGBM | 0.736 | - | No | ~35s |
| **DeBERTa-v3** | **1.000** | **7th** | Yes | ~8.5min |

---

## 🔧 Environment

All code runs on **Kaggle Notebooks** with pre-installed dependencies:
- Python 3.10+
- transformers 4.30+
- scikit-learn 1.3+
- xgboost 2.0+
- lightgbm 4.0+
- torch 2.0+
- pandas, numpy

**No manual installation required** - just click "Run All"!

---

## 📝 Notes

1. **All notebooks are public** and can be accessed directly via the links above
2. **No local setup needed** - everything runs in Kaggle's cloud environment
3. **For best results**: Run Model 3 (DeBERTa-v3) with GPU enabled
4. **Reproducibility**: All models use `random_state=42` for consistent results
5. **Detailed analysis**: See `NLP_ASS.pdf` for complete methodology and discussion

---

## 🏆 Achievement

- DeBERTa-v3 achieved **perfect accuracy (1.000)** on public leaderboard
- Ranked **7th** in the PySphere Movie Review Sentiment Challenge
- Successfully compared traditional ML (Logistic Regression) vs. modern deep learning (DeBERTa-v3)

---

## 📧 Contact

**Student ID**: 25102044g  
**Course**: COMP5423 Natural Language Processing  
**Kaggle Profile**: [moonquakemiao](https://www.kaggle.com/moonquakemiao)

---

**Last Updated**: October 10, 2025
