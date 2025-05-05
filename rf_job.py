#!/usr/bin/env python3
"""
rf_job.py

A standalone PySpark job to:
1. Load preprocessed Random Forest data from Azure Blob Storage (Parquet)
2. Perform hyperparameter tuning with CrossValidator
3. Evaluate on the test set (Accuracy, F1, Precision, Recall, ROC AUC)
4. Save the best model, metrics JSON, and ROC curve plot to Azure Blob Storage
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

def main():
    # ——————————————————————————————————————
    # 1) Spark & Azure Blob Setup
    # ——————————————————————————————————————
    spark = (SparkSession.builder
             .appName("RFJobFullData")
             .getOrCreate())

    
    # Set the access key for your Azure Blob Storage account
    spark.conf.set(
        "fs.azure.account.key.aladdimbigdata.blob.core.windows.net", 
        "access_key_removed"  
    )

    # Versioned output folder
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    container = "wasbs://datafiles@aladdimbigdata.blob.core.windows.net"
    run_base  = f"{container}/rf_model/run_{timestamp}"
    model_path   = f"{run_base}/best_rf_model"
    metrics_path = f"{run_base}/metrics.json"
    roc_blob_path= f"{run_base}/roc_curve.png"
    print(f"🔖 Run base path: {run_base}")

    # ——————————————————————————————————————
    # 2) Load preprocessed data
    # ——————————————————————————————————————
    rf_base = f"{container}/rf_data"
    train_df = spark.read.parquet(f"{rf_base}/train")
    val_df   = spark.read.parquet(f"{rf_base}/val")
    test_df  = spark.read.parquet(f"{rf_base}/test")

    # ——————————————————————————————————————
    # 3) Hyperparameter tuning
    # ——————————————————————————————————————
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="overall", metricName="accuracy")
    evaluator_f1  = MulticlassClassificationEvaluator(labelCol="overall", metricName="f1")
    evaluator_prec= MulticlassClassificationEvaluator(labelCol="overall", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(labelCol="overall", metricName="weightedRecall")

    rf = RandomForestClassifier(featuresCol="final_features", labelCol="overall")
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [40, 60])
                 .addGrid(rf.maxDepth, [5, 10])
                 .build())

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator_acc,
        numFolds=3,
        parallelism=4
    )

    cvModel   = cv.fit(train_df)
    bestModel = cvModel.bestModel

    # ——————————————————————————————————————
    # 4) Save the best model
    # ——————————————————————————————————————
    bestModel.save(model_path)
    print(f"✅ Saved best model to {model_path}")

    # ——————————————————————————————————————
    # 5) Evaluate on test set
    # ——————————————————————————————————————
    preds = bestModel.transform(test_df)

    acc   = evaluator_acc.evaluate(preds)
    f1    = evaluator_f1.evaluate(preds)
    prec  = evaluator_prec.evaluate(preds)
    rec   = evaluator_rec.evaluate(preds)

    # Collect probabilities & labels for ROC
    pdf = preds.select("overall", "probability").toPandas()
    pdf["prob_list"] = pdf["probability"].apply(lambda v: v.toArray().tolist())
    y_true = pdf["overall"].values
    y_score= np.vstack(pdf["prob_list"].values)

    # Trim extra column if model learned 0.0 class
    if y_score.shape[1] == 6:
        y_score = y_score[:,1:]
    classes = [1.0,2.0,3.0,4.0,5.0]
    y_true_bin = label_binarize(y_true, classes=classes)
    roc_auc    = roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")

    # ——————————————————————————————————————
    # 6) Save metrics JSON
    # ——————————————————————————————————————
    metrics = {
        "timestamp": timestamp,
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
        "best_numTrees": bestModel.getNumTrees,
        "best_maxDepth": bestModel.getOrDefault("maxDepth")
    }
    dbutils.fs.put(metrics_path, json.dumps(metrics), overwrite=False)
    print(f"✅ Saved metrics to {metrics_path}")

    # ——————————————————————————————————————
    # 7) Plot & save ROC curve
    # ——————————————————————————————————————
    fpr, tpr, roc_auc_dict = {}, {}, {}
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score[:,i])
        roc_auc_dict[i]   = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {cls} (AUC = {roc_auc_dict[i]:.2f})")
    plt.plot([0,1],[0,1],'k--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest Multiclass ROC")
    plt.legend(loc="lower right")

    # Save locally to DBFS mount, then copy to Blob
    local_path = f"/dbfs/mnt/blob/rf_model/run_{timestamp}/roc_curve.png"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    plt.savefig(local_path)
    print(f"✅ Saved ROC plot to {local_path}")

    

if __name__ == "__main__":
    main()
