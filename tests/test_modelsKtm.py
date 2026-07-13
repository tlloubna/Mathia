import os
import sys

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.calibration import calibration_curve
import src.Process.Estimator.AlphaEstimator as AE
import joblib
def ModelsToTest(has_item_blocks=True):
    """DAS3H : temporalité par CC (tw_kc). DASH : temporalité par item (tw_items)."""
    if has_item_blocks:
        return {
            # users  items  skills wins   fails  attempts  wins_item  attempts_item
            "DAS3H":    {"users": True,  "items": True,  "skills": True,  "wins": True,  "fails": False, "attempts": True,  "wins_item": False, "attempts_item": False},
            "DASH":     {"users": True,  "items": True,  "skills": False, "wins": False, "fails": False, "attempts": False, "wins_item": True,  "attempts_item": True},
            "IRT/MIRT": {"users": True,  "items": True,  "skills": False, "wins": False, "fails": False, "attempts": False, "wins_item": False, "attempts_item": False},
            "PFA":      {"users": False, "items": False, "skills": True,  "wins": True,  "fails": True,  "attempts": False, "wins_item": False, "attempts_item": False},
            "AFM":      {"users": False, "items": False, "skills": True,  "wins": False, "fails": False, "attempts": True,  "wins_item": False, "attempts_item": False},
        }
    # Fallback : pas de blocs item -> DASH approximé par CC
    return {
        "DAS3H":    {"users": True,  "items": True,  "skills": True,  "wins": True,  "fails": False, "attempts": True},
        "DASH":     {"users": True,  "items": True,  "skills": False, "wins": True,  "fails": False, "attempts": True},
        "IRT/MIRT": {"users": True,  "items": True,  "skills": False, "wins": False, "fails": False, "attempts": False},
        "PFA":      {"users": False, "items": False, "skills": True,  "wins": True,  "fails": True,  "attempts": False},
        "AFM":      {"users": False, "items": False, "skills": True,  "wins": False, "fails": False, "attempts": True},
    }

def build_model_columns(models, n_users, n_items, n_skills, n_tw, offset=4, has_item_blocks=True):
    i_u = (offset,  offset + n_users)
    i_i = (i_u[1],  i_u[1] + n_items)
    i_k = (i_i[1],  i_i[1] + n_skills)
    i_w = (i_k[1],  i_k[1] + n_skills * n_tw)
    i_f = (i_w[1],  i_w[1] + n_skills)
    i_a = (i_f[1],  i_f[1] + n_skills * n_tw)

    block_ranges = {
        "users": i_u, "items": i_i, "skills": i_k,
        "wins": i_w, "fails": i_f, "attempts": i_a,
    }
    if has_item_blocks:
        i_wi = (i_a[1],  i_a[1] + n_tw)        
        i_ai = (i_wi[1], i_wi[1] + n_tw)       
        block_ranges["wins_item"] = i_wi
        block_ranges["attempts_item"] = i_ai

    model_cols = {}
    for model_name, blocks in models.items():
        cols = []
        for block, active in blocks.items():
            if active and block in block_ranges:
                start, end = block_ranges[block]
                cols += list(range(start, end))
        model_cols[model_name] = cols
    return model_cols

def make_split_indices(users_col, timestamps, split_mode="user", train_frac=0.8, seed=42):
    np.random.seed(seed)
    train_indices, test_indices = [], []
    if split_mode == "user":
        uu = np.unique(users_col)
        np.random.shuffle(uu)
        sp = int(train_frac * len(uu))
        train_users = set(uu[:sp])
        train_indices = [i for i in range(len(users_col)) if users_col[i] in train_users]
        test_indices  = [i for i in range(len(users_col)) if users_col[i] not in train_users]
    elif split_mode == "interaction":
        for user in np.unique(users_col):
            m = np.where(users_col == user)[0]
            ms = m[np.argsort(timestamps[m])]
            sp = max(1, int(round(train_frac * len(ms))))
            train_indices.extend(ms[:sp].tolist())
            test_indices.extend(ms[sp:].tolist())
    return np.array(train_indices), np.array(test_indices)


def fit_models_mathia(X, n_users, n_items, n_skills, n_tw,
                      split_mode="user", seed=42, has_item_blocks=True):
    y = X[:, 3].toarray().flatten()
    cols_all = list(range(X.shape[1])); cols_all.remove(3)
    Xs = X[:, cols_all]

    users_col = Xs[:, 0].toarray().flatten()
    timestamps = Xs[:, 2].toarray().flatten()
    train_idx, test_idx = make_split_indices(users_col, timestamps, split_mode, 0.8, seed)
    y_train, y_test = y[train_idx], y[test_idx]

    models = ModelsToTest(has_item_blocks=has_item_blocks)
    model_cols = build_model_columns(models, n_users, n_items, n_skills, n_tw,
                                     has_item_blocks=has_item_blocks)

    results = {}
    for name, cols in model_cols.items():
        print(f"\nTraining: {name} ({len(cols)} features)")
        if len(cols) == 0:
            y_pred = np.full(len(y_test), y_train.mean())
        else:
            pipe = Pipeline([
                ("scaler", MaxAbsScaler()),
                ("lr", LogisticRegression(solver="saga", max_iter=500, C=0.1))
            ])
            pipe.fit(Xs[train_idx][:, cols], y_train)
            y_pred = pipe.predict_proba(Xs[test_idx][:, cols])[:, 1]

        results[name] = {
            "AUC": roc_auc_score(y_test, y_pred),
            "NLL": log_loss(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        print(f"  AUC={results[name]['AUC']:.4f} "
              f"NLL={results[name]['NLL']:.4f} "
              f"RMSE={results[name]['RMSE']:.4f}")
    return results
def eval_bayes_mathia(X, n_users, n_items, n_skills, n_tw, seed=42):
    """DAS3H + Bayes : estimation de alpha sur les 100 premières interactions
    de chaque élève de test (split par élève uniquement)."""
    y = X[:, 3].toarray().flatten()
    cols_all = list(range(X.shape[1])); cols_all.remove(3)
    Xs = X[:, cols_all]

    users_col = Xs[:, 0].toarray().flatten()
    timestamps = Xs[:, 2].toarray().flatten()
    train_idx, test_idx = make_split_indices(users_col, timestamps, "user", 0.8, seed)
    y_train, y_test = y[train_idx], y[test_idx]

    # Colonnes DAS3H par CC : users + items + skills + wins + fails + attempts
    offset = 4
    i_a_end = offset + n_users + n_items + n_skills + n_skills*n_tw + n_skills + n_skills*n_tw
    cols = list(range(offset, i_a_end))

    results = {}
    fitBayes(X=Xs, n_users=n_users, Xtrain=Xs[train_idx][:, cols],
             y_train=y_train, test_indices=test_idx, y_test=y_test,
             full_cols=cols, comb="DAS3H+Bayes", results=results)
    return results["DAS3H+Bayes"]


def fitBayes(X, n_users, Xtrain, y_train, test_indices, y_test, full_cols, comb, results):
    pipe = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(solver="saga", max_iter=500, C=0.1))
    ])
    pipe.fit(Xtrain, y_train)
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    coef = lr.coef_[0]
    intercept = lr.intercept_[0]
    alpha_train_coefs = coef[:n_users]
    alpha_nonzero = alpha_train_coefs[alpha_train_coefs != 0]
    prior_mean = np.mean(alpha_nonzero) if len(alpha_nonzero) > 0 else 0.0
    prior_std = np.std(alpha_nonzero)
    params_train = {"pipeline": pipe, "intercept": intercept}
    estimator = AE.alphaEstimator(
        params=params_train, prior_mean=prior_mean,
        prior_std=prior_std, bounds=[-3, 3]
    )
    
    N_init = 100
    X_test_full = X[test_indices][:, full_cols]

    users_test_col = X[test_indices][:, 0].toarray().flatten()
    timestamps_test = X[test_indices][:, 2].toarray().flatten()
    y_pred_comb = np.full(len(y_test), np.nan)
    
    unique_test_users = np.unique(users_test_col)
    print(f"  Bayes: {len(unique_test_users)} élèves test")
    
    for idx, user in enumerate(unique_test_users):
       
        #print(f"    élève {idx}/{len(unique_test_users)}")
        
        user_mask = np.where(users_test_col == user)[0]
        user_sorted = user_mask[np.argsort(timestamps_test[user_mask])]
        X_user = X_test_full[user_sorted]
        
        if len(user_sorted) <= N_init:
            X_scaled = scaler.transform(X_user)
            logits = X_scaled @ coef + intercept + prior_mean
            y_pred_comb[user_sorted] = 1 / (1 + np.exp(-logits))
            continue
        
        X_init = X_user[:N_init]
        y_init = y_test[user_sorted[:N_init]]
        alpha_hat = estimator.estimate_from_X(X_init, y_init)
        
        X_eval_scaled = scaler.transform(X_user[N_init:])
        logits = X_eval_scaled @ coef + intercept + alpha_hat
        y_pred_comb[user_sorted[N_init:]] = 1 / (1 + np.exp(-logits))
        
        X_init_scaled = scaler.transform(X_init)
        logits_init = X_init_scaled @ coef + intercept + alpha_hat
        y_pred_comb[user_sorted[:N_init]] = 1 / (1 + np.exp(-logits_init))
    
    valid = ~np.isnan(y_pred_comb)
    y_pred_comb = y_pred_comb[valid]
    y_test_bayes = y_test[valid]
    results[comb] = {
        "AUC": roc_auc_score(y_test_bayes, y_pred_comb),
        "NLL": log_loss(y_test_bayes, y_pred_comb),
        "RMSE": np.sqrt(mean_squared_error(y_test_bayes, y_pred_comb)),
        "y_test": y_test_bayes,
        "y_pred": y_pred_comb,
    }
    print(f"  AUC: {results[comb]['AUC']:.4f}, NLL: {results[comb]['NLL']:.4f}")


def print_metric_table(results, split_mode):
    print(f"\n=== MATHIA — split {split_mode} ===")
    print(f"{'Modèle':<12}{'AUC':>9}{'NLL':>9}{'RMSE':>9}")
    print("-" * 39)
    for name, r in results.items():
        print(f"{name:<12}{r['AUC']:>9.4f}{r['NLL']:>9.4f}{r['RMSE']:>9.4f}")


if __name__ == "__main__":
   # ("ASSISTments13_12", 15698)
   # ("algebra05", 574)
    #("bridge_algebra06", 1146)
    #("Mathiadata", 25351),
    #("ASSISTments13_12", 15698),
    N_TW = 5
    N_STUDENTS = 574
    data_folder = os.path.join("data", "algebra05")

    path_dash = os.path.join(data_folder, f"history_features_DASH_{N_STUDENTS}std.npz")
    path_meta_dash = os.path.join(data_folder, f"history_metadata_DASH_{N_STUDENTS}std.npz")
    path_std = os.path.join(data_folder, f"history_features_{N_STUDENTS}std.npz")
    path_meta_std = os.path.join(data_folder, f"history_metadata_{N_STUDENTS}std.npz")
    
    if os.path.exists(path_dash):
        X = sparse.load_npz(path_dash)
        metadata = np.load(path_meta_dash, allow_pickle=True)
        print(">>> Matrice AVEC blocs item chargée (vrai DASH possible)")
    else:
        X = sparse.load_npz(path_std)
        metadata = np.load(path_meta_std, allow_pickle=True)
        print(">>> Matrice SANS blocs item (DASH = approximation par CC)")

    n_users = len(metadata["user_ids"])
    n_items = len(metadata["item_ids"])
    n_kcs   = len(metadata["kc_list"])

    """# Détection automatique des blocs item
    base = 4 + n_users + n_items + n_kcs + n_kcs*N_TW + n_kcs + n_kcs*N_TW
    has_item_blocks = (X.shape[1] >= base + 2*(n_items*N_TW))
    print(f"X.shape={X.shape}  has_item_blocks={has_item_blocks}")"""

    # --- Scénario 1 : split par élève (cold start) ---
    res_user = fit_models_mathia(X, n_users, n_items, n_kcs, N_TW,
                                 split_mode="user", seed=42,
                                 has_item_blocks=True)
    print_metric_table(res_user, "user")

    # --- Scénario 2 : split par interaction (élève connu) ---
    res_inter = fit_models_mathia(X, n_users, n_items, n_kcs, N_TW,
                                  split_mode="interaction", seed=42,
                                  has_item_blocks=True)
    print_metric_table(res_inter, "interaction")

    print("\n!!! done !!!")