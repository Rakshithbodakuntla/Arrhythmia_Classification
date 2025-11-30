import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from wfdb import rdrecord, rdann
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table

# ✅ Hardcoded data directory
DATA_DIR = r"C:\Users\raksh\Desktop\Project1\arrythmia\arrhythmia_classification-master\Experiments\data\mitdb"
WIN_SAMPLES = 180

# --- ECG Data Loader ---
def load_ecg_data(data_dir=DATA_DIR, win_samples=WIN_SAMPLES):
    X, y = [], []
    recs = [f.split(".")[0] for f in os.listdir(data_dir) if f.endswith(".dat")]
    print(f"📥 Found {len(recs)} ECG records in {data_dir}")
    for r in recs:
        try:
            record = rdrecord(os.path.join(data_dir, r))
            ann = rdann(os.path.join(data_dir, r), 'atr')
            sig = record.p_signal[:, 0]
            beats = ann.sample
            labels = ann.symbol
            for i, b in enumerate(beats):
                start = max(b - win_samples // 2, 0)
                end = min(b + win_samples // 2, len(sig))
                seg = np.zeros(win_samples)
                seg[:end - start] = sig[start:end]
                X.append(seg)
                y.append(labels[i])
        except Exception as e:
            print(f"⚠️ Skipped {r}: {e}")
    X = np.array(X).reshape(-1, win_samples, 1)
    y = np.array(y)
    return X, y

# --- TSFRESH Feature Ranking ---
def compute_and_save_tsfresh_ranking(X, y, results_dir="results", sample_cap=2000):
    os.makedirs(results_dir, exist_ok=True)
    n = min(sample_cap, X.shape[0])
    idx = np.random.choice(X.shape[0], n, replace=False)
    Xs, ys = X[idx, :, 0], y[idx]
    df_list = [pd.DataFrame({"id": i, "time": np.arange(Xs.shape[1]), "val": seq}) for i, seq in enumerate(Xs)]
    df_long = pd.concat(df_list)
    feats = extract_features(df_long, column_id="id", column_sort="time",
                             default_fc_parameters=MinimalFCParameters(), n_jobs=0)
    impute(feats)
    y_series = pd.Series(ys, index=feats.index)
    rel = calculate_relevance_table(feats, y_series).sort_values("p_value")
    rel.to_csv(os.path.join(results_dir, "tsfresh_relevance.csv"), index=False)
    print(f"✅ TSFRESH extracted {feats.shape[1]} features — relevance saved.")
    return rel

# --- Dataset Builder ---
def build_dataset():
    X, y = load_ecg_data(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(-1, WIN_SAMPLES, 1)
    X_rs, y_rs = RandomOverSampler().fit_resample(X.reshape(X.shape[0], -1), y)
    X = X_rs.reshape(-1, WIN_SAMPLES, 1)
    y = y_rs
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print(f"✅ Built dataset — Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test
