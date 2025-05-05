# Amazon Review Rating Prediction

Predicted Amazon product review star-ratings using PySpark, Word2Vec embeddings, and classical ML models on Azure Databricks.

## 🚀 Features
- Generates Word2Vec embeddings of review text at scale using PySpark
- Cleans and preprocesses raw review text at scale using PySpark (tokenization, stop-word removal, TF‑IDF)
- Trains and compares multiple classifiers:
  - Logistic Regression (using Word2Vec embeddings)
  - Random Forest (using TF‑IDF features)  
- Performs hyperparameter tuning via grid search and cross‑validation
- Evaluates models on accuracy, F1-score, precision, recall, and ROC AUC
- Generates visual performance reports

## 🛠️ Tech Stack
- **Languages & Frameworks:** Python, PySpark, Jupyter Notebook
- **Cloud & Big Data:** Azure Databricks (2 jobs), Spark, Azure Blob Storage
- **Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## ⚙️ Workflow
1. **Data Preprocessing Job** (Databricks Job #1): PySpark notebook for cleaning, TF‑IDF feature extraction, and saving processed data to DBFS.
2. **Model Training Job** (Databricks Job #2): PySpark notebook to train Logistic Regression and Random Forest, perform hyperparameter tuning, and serialize models to DBFS.
3. Local notebooks for EDA and visualization; uses model outputs to generate performance charts.

## 📥 Installation & Usage
1. Clone this repo and `cd AmazonReviewRating`
2. Configure Databricks CLI and upload notebooks to your workspace
3. Submit Databricks jobs via UI or CLI; outputs saved to `/dbfs/models/`
4. Download models locally and score new reviews:
   ```bash
   python lg_job.py --model models/lg.pkl --reviews "Great product, loved it!"


## 📈 Results
| Model               | Accuracy | F1-Score | Precision | Recall | ROC AUC | Best Params                 |
|---------------------|----------|----------|-----------|--------|---------|-----------------------------|
| Logistic Regression | 45.19%   | 43.62%   | 43.71%    | 45.19% | 0.7586  | regParam=0.0, maxIter=50    |
| Random Forest       | 38.51%   | 36.77%   | 37.34%    | 38.51% | 0.7089  | numTrees=60, maxDepth=10    |



