"""
Run all models sequentially, compute metrics, and save graphs.
"""

import os, matplotlib.pyplot as plt, pandas as pd
from data import build_dataset, compute_and_save_tsfresh_ranking, WIN_SAMPLES
from models import build_all_models
from train_eval import train_one, evaluate_model

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_history(hist, model_name):
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title(f"{model_name} Accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.legend(); plt.title(f"{model_name} Loss")
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_loss.png"))
    plt.close()

def main():
    print("📥 Loading ECG data and building dataset...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = build_dataset()
    compute_and_save_tsfresh_ranking(X_tr, y_tr, results_dir=RESULTS_DIR)

    input_shape = (WIN_SAMPLES, 1)
    num_classes = len(set(y_tr))
    models = build_all_models(input_shape, num_classes)
    all_results = []

    for name, model in models.items():
        print(f"\n🚀 Training {name} ...")
        hist, runtime, cpu = train_one(model, X_tr, y_tr, X_va, y_va, RESULTS_DIR, epochs=15)
        plot_history(hist, name)
        metrics = evaluate_model(model, X_te, y_te, RESULTS_DIR)
        metrics.update({"model": name, "runtime": runtime, "cpu": cpu})
        all_results.append(metrics)
        print(f"✅ {name}: Accuracy {metrics['accuracy']:.4f}")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "all_metrics.csv"), index=False)
    print("\n📊 Summary saved to all_metrics.csv")

if __name__ == "__main__":
    main()
