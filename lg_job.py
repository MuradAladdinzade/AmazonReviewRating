

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

def main():
    # -----------------------------------------------------------
    # 1) Spark Session & Paths
    # -----------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("LRJobFullData")
        .getOrCreate()
    )
    # Set the access key for your Azure Blob Storage account
    spark.conf.set(
        "fs.azure.account.key.aladdimbigdata.blob.core.windows.net", 
        "access_key_removed"  
    )

    # Unique run folder
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    container = "wasbs://datafiles@aladdimbigdata.blob.core.windows.net"
    run_base  = f"{container}/lr_model/run_{timestamp}"  # <== changed from 'rf_model'
    model_path   = f"{run_base}/best_lr_model"
    metrics_path = f"{run_base}/metrics.json"
    roc_blob_path= f"{run_base}/roc_curve.png"
    print(f"🔖 Run base path: {run_base}")

    # -----------------------------------------------------------
    # 2) Load Parquet data (train/val/test)
    #    Must include 'final_features' and 'overall'
    # -----------------------------------------------------------
    lr_base = f"{container}/rf_data"  
    # ^ If your LR data is in a different folder, change it here.
    #   For now we assume it's the same 'rf_data' location
    #   just containing final_features + overall.

    train_df = spark.read.parquet(f"{lr_base}/train")
    val_df   = spark.read.parquet(f"{lr_base}/val")
    test_df  = spark.read.parquet(f"{lr_base}/test")

    # -----------------------------------------------------------
    # 3) Define Logistic Regression + CrossValidator
    # -----------------------------------------------------------
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="overall", metricName="accuracy")
    evaluator_f1  = MulticlassClassificationEvaluator(labelCol="overall", metricName="f1")
    evaluator_prec= MulticlassClassificationEvaluator(labelCol="overall", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(labelCol="overall", metricName="weightedRecall")

    # Logistic Regression
    lr = LogisticRegression(
        featuresCol="final_features",
        labelCol="overall"
    )

    # Hyperparam grid (example: regParam + maxIter)
    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0])
        .addGrid(lr.maxIter, [50, 100, 120])
        .build()
    )

    # CrossValidator
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator_acc,  # optimize for accuracy
        numFolds=3,
        parallelism=4
    )

    # -----------------------------------------------------------
    # 4) Fit on train data & pick best model
    # -----------------------------------------------------------
    cvModel   = cv.fit(train_df)
    bestModel = cvModel.bestModel
    print("✅ Best Logistic Regression Model found with params:")
    print(f"   • regParam  = {bestModel.getRegParam()}")
    print(f"   • maxIter   = {bestModel.getMaxIter()}")

    # Save best model
    bestModel.save(model_path)
    print(f"✅ Saved best model to {model_path}")

    # -----------------------------------------------------------
    # 5) Evaluate on test set
    # -----------------------------------------------------------
    preds = bestModel.transform(test_df)

    acc   = evaluator_acc.evaluate(preds)
    f1    = evaluator_f1.evaluate(preds)
    prec  = evaluator_prec.evaluate(preds)
    rec   = evaluator_rec.evaluate(preds)

    # -----------------------------------------------------------
    # 6) Compute ROC AUC (multiclass)
    # -----------------------------------------------------------
    pdf = preds.select("overall", "probability").toPandas()
    pdf["prob_list"] = pdf["probability"].apply(lambda v: v.toArray().tolist())
    y_true = pdf["overall"].values
    y_score= np.vstack(pdf["prob_list"].values)

    # If the model outputs 6 columns for some reason, drop the first col
    if y_score.shape[1] == 6:
        print("⚠️ Found 6 columns in probability vector; trimming first column.")
        y_score = y_score[:,1:]

    classes = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_true_bin = label_binarize(y_true, classes=classes)
    roc_auc    = roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")

    # -----------------------------------------------------------
    # 7) Save metrics to JSON
    # -----------------------------------------------------------
    import json
    metrics = {
        "timestamp": timestamp,
        "accuracy":  float(acc),
        "f1_score":  float(f1),
        "precision": float(prec),
        "recall":    float(rec),
        "roc_auc":   float(roc_auc),
        "best_regParam": bestModel.getRegParam(),
        "best_maxIter":  bestModel.getMaxIter()
    }
    dbutils.fs.put(metrics_path, json.dumps(metrics), overwrite=False)
    print(f"✅ Saved metrics to {metrics_path}")

    # -----------------------------------------------------------
    # 8) Plot & save ROC curve
    # -----------------------------------------------------------
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, roc_auc_dict = {}, {}, {}
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_dict[i]   = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {cls} (AUC={roc_auc_dict[i]:.2f})")

    plt.plot([0,1], [0,1], 'k--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression Multiclass ROC")
    plt.legend(loc="lower right")

    # Save locally to DBFS, then it’s in Blob
    local_path = f"/dbfs/mnt/blob/lr_model/run_{timestamp}/roc_curve.png"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    plt.savefig(local_path)
    print(f"✅ Saved ROC plot to {local_path}")

    print("✅ Done with LR job!")

if __name__ == "__main__":
    main()
