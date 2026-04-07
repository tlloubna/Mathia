import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import os, sys
import numpy as np

import src.Process.DAS3H as DAS3H
import src.datamodel.Historydata as HIS
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt 
from collections import defaultdict
import random
NAME_FOLDER  = "simulated"
DATA_FOLDER  = os.path.join("data", NAME_FOLDER)
N_STUDENTS   = 1146
MAX_GAP_DAYS = 7
MAX_SESSIONS = 10  # on trace jusqu'à 10 séances
def evaluate_next_step(df, q_matrix, windows, sid=0, min_train=10):
    df_s = df[df["user_id"] == sid].sort_values("timestamp").reset_index(drop=True)
    n    = len(df_s)
    
    predictions = []
    
    for t in range(min_train, n):
        df_train = df_s.iloc[:t]
        df_test  = df_s.iloc[t:t+1]
        
        if len(df_train["correct"].unique()) < 2:
            continue
        
        vocab_users = sorted(set(df_train["user_id"]) | set(df_test["user_id"]))
        vocab_items = sorted(set(df_train["item_id"]) | set(df_test["item_id"]))
        
        his = HIS.HistoryDATA(TimeWindow=windows)
        X_train, user_train, item_train, _ = his.ComputeHistoryFeaturesTWKC(
            q_matrix, df_train,
            vocab_users=vocab_users, vocab_items=vocab_items
        )
        X_test, _, _, _ = his.ComputeHistoryFeaturesTWKC(
            q_matrix, df_test,
            vocab_users=vocab_users, vocab_items=vocab_items
        )
        
        kc_list = [str(i) for i in range(q_matrix.shape[1])]
        model   = DAS3H.DAS3HModel(C=1.0)
        
        # entraîner manuellement sans calculer AUC
        y_train = X_train[:, 3].toarray().flatten()
        y_test  = X_test[:, 3].toarray().flatten()
        cols    = [c for c in range(X_train.shape[1]) if c != 3]
        
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.linear_model import LogisticRegression
        
        pipe = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("lr", LogisticRegression(solver="saga", max_iter=2000, C=1.0))
        ])
        pipe.fit(X_train[:, cols], y_train)
        
        # prédire directement sans AUC
        y_pred = pipe.predict_proba(X_test[:, cols])[:, 1]
        
        predictions.append({
            "t":      t,
            "y_pred": float(y_pred[0]),
            "y_true": float(y_test[0]),
            "brier_t": (float(y_pred[0]) - float(y_test[0])) ** 2
        })
    
    if len(predictions) == 0:
        return None
    
    df_pred = pd.DataFrame(predictions)
    brier   = df_pred["brier_t"].mean()
    accuracy = ((df_pred["y_pred"] > 0.5) == df_pred["y_true"]).mean()
    
    df_pred["brier_rolling"] = df_pred["brier_t"].rolling(10, min_periods=1).mean()
    
    print(f"Étudiant {sid} — {len(predictions)} prédictions")
    print(f"Brier Score : {brier:.4f}  (0=parfait, 0.25=aléatoire)")
    print(f"Accuracy    : {accuracy:.4f}")
    
    return df_pred

def plot_next_step(df_pred_by_student: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_pred_by_student)))
    
    for (sid, df_pred), color in zip(df_pred_by_student.items(), colors):
        if df_pred is None:
            continue
        # Brier glissant par étudiant
        ax1.plot(df_pred["t"], df_pred["brier_rolling"],
                 linewidth=1, alpha=0.6, color=color, label=f"Étudiant {sid}")
    
    ax1.axhline(0.25, linestyle="--", color="gray", linewidth=1, label="aléatoire")
    ax1.set_ylabel("Brier Score glissant ↓")
    ax1.set_xlabel("Interaction t")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_title("Évolution du Brier Score au fil des interactions")
    
    # Brier moyen par position t (agrégé sur tous les étudiants)
    from collections import defaultdict
    brier_by_t = defaultdict(list)
    for df_pred in df_pred_by_student.values():
        if df_pred is None:
            continue
        for _, row in df_pred.iterrows():
            brier_by_t[row["t"]].append(row["brier_t"])
    
    ts      = sorted(brier_by_t.keys())
    means   = [np.mean(brier_by_t[t]) for t in ts]
    
    ax2.plot(ts, means, color="steelblue", linewidth=2)
    ax2.axhline(0.25, linestyle="--", color="gray", linewidth=1, label="aléatoire")
    ax2.set_ylabel("Brier Score moyen ↓")
    ax2.set_xlabel("Interaction t")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Brier Score moyen agrégé sur tous les étudiants")
    
    plt.suptitle("Évaluation next-step de DAS3H — Brier Score", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df       = pd.read_csv(os.path.join(DATA_FOLDER, "preprocessed_data_simulated.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, "q_mat_simulated.npz")).toarray()
    windows  = [3600, 3600*24, 3600*24*7, 3600*24*30, float("inf")]
    
    students = sorted(df["user_id"].unique())
    df_pred_by_student = {}
    
    for sid in students[:5]:  # tester sur 5 étudiants
        print(f"\nÉtudiant {sid}...")
        df_pred = evaluate_next_step(df, q_matrix, windows, sid=sid, min_train=10)
        df_pred_by_student[sid] = df_pred
    
    plot_next_step(df_pred_by_student)
print("!!!!!!!!!!done!!!!!!!!!!")