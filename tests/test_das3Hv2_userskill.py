import os
import sys

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

# On réutilise les fonctions de ton fichier de test principal
from test_modelsKtm import make_split_indices, eval_bayes_mathia


def eval_das3h_userskill(data_folder, n_students, split_mode="user", seed=42):
    """DAS3H user/skill (variante Alpha) réentraîné sur le MÊME split,
    à partir de la matrice Alpha history_features_Alpha{N}std.npz."""
    path = os.path.join(data_folder, f"history_features_Alpha{n_students}std.npz")
    if not os.path.exists(path):
        print(f"[!] Matrice Alpha introuvable : {path}")
        return None

    Xa = sparse.load_npz(path)
    y = Xa[:, 3].toarray().flatten()
    cols_all = list(range(Xa.shape[1])); cols_all.remove(3)
    Xs = Xa[:, cols_all]

    users_col = Xs[:, 0].toarray().flatten()
    timestamps = Xs[:, 2].toarray().flatten()
    train_idx, test_idx = make_split_indices(users_col, timestamps, split_mode, 0.8, seed)
    y_train, y_test = y[train_idx], y[test_idx]

    # Toutes les features de la matrice Alpha (hors les 4 colonnes métadonnées)
    cols = list(range(4, Xs.shape[1]))

    pipe = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(solver="saga", max_iter=500, C=0.1))
    ])
    pipe.fit(Xs[train_idx][:, cols], y_train)
    y_pred = pipe.predict_proba(Xs[test_idx][:, cols])[:, 1]

    return {
        "AUC": roc_auc_score(y_test, y_pred),
        "NLL": log_loss(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def print_table(results, title):
    print(f"\n=== {title} ===")
    print(f"{'Modèle':<22}{'AUC':>9}{'NLL':>9}{'RMSE':>9}")
    print("-" * 49)
    for name, r in results.items():
        if r is None:
            print(f"{name:<22}{'--':>9}{'--':>9}{'--':>9}")
        else:
            print(f"{name:<22}{r['AUC']:>9.4f}{r['NLL']:>9.4f}{r['RMSE']:>9.4f}")


if __name__ == "__main__":
    # ("algebra05", 574)
    #("bridge_algebra06", 1146)
    #("Mathiadata", 25351),
    #("ASSISTments13_12", 15698),
    N_TW = 5
    N_STUDENTS = 574
    FOLDER = "algebra05"
    data_folder = os.path.join("data", FOLDER)
    path_std = os.path.join(data_folder, f"history_features_{N_STUDENTS}std.npz")
    path_meta = os.path.join(data_folder, f"history_metadata_{N_STUDENTS}std.npz")
    X = sparse.load_npz(path_std)
    metadata = np.load(path_meta, allow_pickle=True)

    n_users = len(metadata["user_ids"])
    n_items = len(metadata["item_ids"])
    n_kcs   = len(metadata["kc_list"])
    print(f"X.shape={X.shape}  n_users={n_users} n_items={n_items} n_kcs={n_kcs}")

    results = {}

    print("\n>>> DAS3H + Bayes (split user)")
    try:
        results["DAS3H+Bayes (user)"] = eval_bayes_mathia(
            X, n_users, n_items, n_kcs, N_TW, seed=42
        )
    except Exception as e:
        print(f"[!] Bayes échoué : {e}")
        results["DAS3H+Bayes (user)"] = None

    # --- DAS3H user/skill, dans les deux splits ---
    print("\n>>> DAS3H user/skill (split user)")
    results["DAS3H user/skill (user)"] = eval_das3h_userskill(
        data_folder, N_STUDENTS, split_mode="user", seed=42)
    print("\n>>> DAS3H user/skill (split interaction)")
    results["DAS3H user/skill (inter)"] = eval_das3h_userskill(
        data_folder, N_STUDENTS, split_mode="interaction", seed=42
    )

    print_table(results, f"{FOLDER} — variantes DAS3H")
    print("\n!!! done !!!")