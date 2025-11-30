import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# --------------------------------------------------
# Train one model
# --------------------------------------------------
def train_one(model, X_tr, y_tr, X_va, y_va, results_dir, epochs=10, batch_size=64):
    start_time = time.time()

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(results_dir, f"{model.name}_best.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    runtime = time.time() - start_time
    cpu_usage = 0.0  # placeholder (optional psutil integration)
    return history, runtime, cpu_usage


# --------------------------------------------------
# Evaluate model with extended metrics + confusion matrix
# --------------------------------------------------
def evaluate_model(model, X_te, y_te, results_dir=None):
    """Evaluate model and save precision, recall, F1, AUC, confusion matrix."""
    print(f"\n📊 Evaluating model: {model.name}")
    y_pred_prob = model.predict(X_te, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_te, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_te, y_pred),
        "kappa": cohen_kappa_score(y_te, y_pred)
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            tf.one_hot(y_te, y_pred_prob.shape[1]), y_pred_prob, multi_class="ovr"
        )
    except Exception:
        metrics["roc_auc"] = np.nan

    print(f"✅ Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    # Plot confusion matrix (IEEE-style)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model.name}")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_te)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Cell text values
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        save_path = os.path.join(results_dir, f"{model.name}_confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Saved confusion matrix: {save_path}")

    return metrics
