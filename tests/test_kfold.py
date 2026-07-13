

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
import src.Process.Estimator.AlphaEstimator as AE


# ============================================================
# 1. Définition des modèles
# ============================================================
def ModelsToTest(has_item_blocks=True):
    """DAS3H : temporalité par CC (tw_kc). DASH : temporalité par item (tw_items)."""
    if has_item_blocks:
        return {
            "DAS3H":    {"users": True,  "items": True,  "skills": True,  "wins": True,  "fails": False, "attempts": True,  "wins_item": False, "attempts_item": False},
            "DASH":     {"users": True,  "items": True,  "skills": False, "wins": False, "fails": False, "attempts": False, "wins_item": True,  "attempts_item": True},
            "IRT/MIRT": {"users": True,  "items": True,  "skills": False, "wins": False, "fails": False, "attempts": False, "wins_item": False, "attempts_item": False},
            "PFA":      {"users": False, "items": False, "skills": True,  "wins": True,  "fails": True,  "attempts": False, "wins_item": False, "attempts_item": False},
            "AFM":      {"users": False, "items": False, "skills": True,  "wins": False, "fails": False, "attempts": True,  "wins_item": False, "attempts_item": False},
        }
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



def make_cv_folds(users_col, timestamps, split_mode="user", n_folds=5, seed=42):
    
    folds = []

    if split_mode == "user":
        rng = np.random.RandomState(seed)
        unique_users = np.unique(users_col)
        rng.shuffle(unique_users)
        user_chunks = np.array_split(unique_users, n_folds)
        for k in range(n_folds):
            test_users = set(user_chunks[k].tolist())
            test_idx  = np.array([i for i in range(len(users_col)) if users_col[i] in test_users])
            train_idx = np.array([i for i in range(len(users_col)) if users_col[i] not in test_users])
            folds.append((train_idx, test_idx))

    elif split_mode == "interaction":
        # Pli k = k-ième tranche chronologique de 1/n_folds pour chaque élève.
        # (partition temporelle : chaque interaction est testée exactement une fois)
        for k in range(n_folds):
            train_idx, test_idx = [], []
            for user in np.unique(users_col):
                m = np.where(users_col == user)[0]
                ms = m[np.argsort(timestamps[m])]
                n = len(ms)
                if n < n_folds:
                    train_idx.extend(ms.tolist())
                    continue
                bounds = np.linspace(0, n, n_folds + 1).astype(int)
                lo, hi = bounds[k], bounds[k + 1]
                test_idx.extend(ms[lo:hi].tolist())
                train_idx.extend(np.concatenate([ms[:lo], ms[hi:]]).tolist())
            folds.append((np.array(train_idx), np.array(test_idx)))

    return folds



def fit_models_cv(X, n_users, n_items, n_skills, n_tw,
                  split_mode="user", n_folds=5, seed=42, has_item_blocks=True):
    y = X[:, 3].toarray().flatten()
    cols_all = list(range(X.shape[1])); cols_all.remove(3)
    Xs = X[:, cols_all]
    users_col = Xs[:, 0].toarray().flatten()
    timestamps = Xs[:, 2].toarray().flatten()

    models = ModelsToTest(has_item_blocks=has_item_blocks)
    model_cols = build_model_columns(models, n_users, n_items, n_skills, n_tw,
                                     has_item_blocks=has_item_blocks)

    folds = make_cv_folds(users_col, timestamps, split_mode, n_folds, seed)
    raw = {name: {"AUC": [], "NLL": [], "RMSE": []} for name in model_cols}

    for k, (train_idx, test_idx) in enumerate(folds):
        print(f"\n----- Pli {k+1}/{n_folds} ({split_mode}) "
              f"| train={len(train_idx)} test={len(test_idx)} -----")
        y_train, y_test = y[train_idx], y[test_idx]

        for name, cols in model_cols.items():
            if len(cols) == 0:
                y_pred = np.full(len(y_test), y_train.mean())
            else:
                pipe = Pipeline([
                    ("scaler", MaxAbsScaler()),
                    ("lr", LogisticRegression(solver="saga", max_iter=500, C=0.1))
                ])
                pipe.fit(Xs[train_idx][:, cols], y_train)
                y_pred = pipe.predict_proba(Xs[test_idx][:, cols])[:, 1]

            raw[name]["AUC"].append(roc_auc_score(y_test, y_pred))
            raw[name]["NLL"].append(log_loss(y_test, y_pred))
            raw[name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            print(f"  {name:<10} AUC={raw[name]['AUC'][-1]:.4f}")

    summary = {name: {m: (np.mean(raw[name][m]), np.std(raw[name][m]))
                      for m in ("AUC", "NLL", "RMSE")}
               for name in model_cols}
    return summary, raw

def fitBayes(X, n_users, Xtrain, y_train, test_indices, y_test, full_cols, comb, results,
             N_init=100):
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
    estimator = AE.alphaEstimator(
        params={"pipeline": pipe, "intercept": intercept},
        prior_mean=prior_mean, prior_std=prior_std, bounds=[-3, 3]
    )

    X_test_full = X[test_indices][:, full_cols]
    users_test_col = X[test_indices][:, 0].toarray().flatten()
    timestamps_test = X[test_indices][:, 2].toarray().flatten()
    y_pred_comb = np.full(len(y_test), np.nan)

    for user in np.unique(users_test_col):
        mask = np.where(users_test_col == user)[0]
        srt = mask[np.argsort(timestamps_test[mask])]
        X_user = X_test_full[srt]

        if len(srt) <= N_init:
            X_scaled = scaler.transform(X_user)
            logits = X_scaled @ coef + intercept + prior_mean
            y_pred_comb[srt] = 1 / (1 + np.exp(-logits))
            continue

        X_init = X_user[:N_init]
        y_init = y_test[srt[:N_init]]
        alpha_hat = estimator.estimate_from_X(X_init, y_init)

        X_eval = scaler.transform(X_user[N_init:])
        y_pred_comb[srt[N_init:]] = 1 / (1 + np.exp(-(X_eval @ coef + intercept + alpha_hat)))
        X_init_s = scaler.transform(X_init)
        y_pred_comb[srt[:N_init]] = 1 / (1 + np.exp(-(X_init_s @ coef + intercept + alpha_hat)))

    valid = ~np.isnan(y_pred_comb)
    yp, yt = y_pred_comb[valid], y_test[valid]
    results[comb] = {
        "AUC": roc_auc_score(yt, yp),
        "NLL": log_loss(yt, yp),
        "RMSE": np.sqrt(mean_squared_error(yt, yp)),
    }


def eval_bayes_cv(X, n_users, n_items, n_skills, n_tw, n_folds=5, seed=42):
    y = X[:, 3].toarray().flatten()
    cols_all = list(range(X.shape[1])); cols_all.remove(3)
    Xs = X[:, cols_all]
    users_col = Xs[:, 0].toarray().flatten()
    timestamps = Xs[:, 2].toarray().flatten()

    offset = 4
    i_a_end = offset + n_users + n_items + n_skills + n_skills*n_tw + n_skills + n_skills*n_tw
    cols = list(range(offset, i_a_end))

    folds = make_cv_folds(users_col, timestamps, "user", n_folds, seed)
    raw = {"AUC": [], "NLL": [], "RMSE": []}

    for k, (train_idx, test_idx) in enumerate(folds):
        print(f"\n----- Bayes pli {k+1}/{n_folds} -----")
        res = {}
        fitBayes(X=Xs, n_users=n_users, Xtrain=Xs[train_idx][:, cols],
                 y_train=y[train_idx], test_indices=test_idx, y_test=y[test_idx],
                 full_cols=cols, comb="DAS3H+Bayes", results=res)
        for m in ("AUC", "NLL", "RMSE"):
            raw[m].append(res["DAS3H+Bayes"][m])
        print(f"  AUC={raw['AUC'][-1]:.4f}")

    return {m: (np.mean(raw[m]), np.std(raw[m])) for m in raw}



def eval_userskill_cv(data_folder, n_students, n_tw,
                      split_mode="user", n_folds=5, seed=42):
    path = os.path.join(data_folder, f"history_features_Alpha{n_students}std.npz")
    if not os.path.exists(path):
        print(f"  [!] Matrice Alpha introuvable : {path}")
        return None

    Xa = sparse.load_npz(path)
    y = Xa[:, 3].toarray().flatten()
    cols_all = list(range(Xa.shape[1])); cols_all.remove(3)
    Xs = Xa[:, cols_all]
    users_col = Xs[:, 0].toarray().flatten()
    timestamps = Xs[:, 2].toarray().flatten()
    cols = list(range(4, Xs.shape[1]))

    folds = make_cv_folds(users_col, timestamps, split_mode, n_folds, seed)
    raw = {"AUC": [], "NLL": [], "RMSE": []}

    for k, (train_idx, test_idx) in enumerate(folds):
        pipe = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("lr", LogisticRegression(solver="saga", max_iter=500, C=0.1))
        ])
        pipe.fit(Xs[train_idx][:, cols], y[train_idx])
        y_pred = pipe.predict_proba(Xs[test_idx][:, cols])[:, 1]
        raw["AUC"].append(roc_auc_score(y[test_idx], y_pred))
        raw["NLL"].append(log_loss(y[test_idx], y_pred))
        raw["RMSE"].append(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
        print(f"  user/skill pli {k+1}/{n_folds} AUC={raw['AUC'][-1]:.4f}")

    return {m: (np.mean(raw[m]), np.std(raw[m])) for m in raw}


def print_cv_table(summary, split_mode, dataset=""):
    print(f"\n=== {dataset} — CV {split_mode} (moyenne ± écart-type) ===")
    print(f"{'Modèle':<16}{'AUC':>18}{'NLL':>18}{'RMSE':>18}")
    print("-" * 70)
    for name, r in summary.items():
        auc  = f"{r['AUC'][0]:.4f}±{r['AUC'][1]:.4f}"
        nll  = f"{r['NLL'][0]:.4f}±{r['NLL'][1]:.4f}"
        rmse = f"{r['RMSE'][0]:.4f}±{r['RMSE'][1]:.4f}"
        print(f"{name:<16}{auc:>18}{nll:>18}{rmse:>18}")


def print_full_cv_table(results_user, results_inter, dataset="", n_folds=5):
    
    order = ["DAS3H+Bayes", "DAS3H user/skill", "DAS3H", "DASH", "IRT/MIRT", "PFA", "AFM"]
    def fmt(stats, metric):
        if stats is None:
            return "---"
        m, s = stats[metric]
        return f"{m:.4f}±{s:.4f}"

    print(f"\n{'='*100}")
    print(f"  {dataset} — Validation croisée {n_folds}-fold (moyenne ± écart-type)")
    print(f"{'='*100}")
    # En-tête à deux niveaux
    print(f"{'':<18}{'SPLIT PAR ÉLÈVE':^39}|{'SPLIT PAR INTERACTION':^39}")
    print(f"{'Modèle':<18}{'AUC':>13}{'NLL':>13}{'RMSE':>13} |"
          f"{'AUC':>13}{'NLL':>13}{'RMSE':>13}")
    print("-" * 100)

    for name in order:
        ru = results_user.get(name)
        ri = results_inter.get(name)
        # Si le modèle est totalement absent des deux, on saute
        if ru is None and ri is None:
            continue
        print(f"{name:<18}"
              f"{fmt(ru,'AUC'):>13}{fmt(ru,'NLL'):>13}{fmt(ru,'RMSE'):>13} |"
              f"{fmt(ri,'AUC'):>13}{fmt(ri,'NLL'):>13}{fmt(ri,'RMSE'):>13}")
    print("=" * 100)


if __name__ == "__main__":
    # ("algebra05", 574)
    #("bridge_algebra06", 1146)
    #("Mathiadata", 25351),
    #("ASSISTments13_12", 15698),
    N_TW = 5
    N_FOLDS = 5
    SEED = 42
    N_STUDENTS = 25351
    FOLDER = "Mathiadata"
    data_folder = os.path.join("data", FOLDER)

    path_dash = os.path.join(data_folder, f"history_features_DASH_{N_STUDENTS}std.npz")
    path_meta_dash = os.path.join(data_folder, f"history_metadata_DASH_{N_STUDENTS}std.npz")
    if os.path.exists(path_dash):
        X = sparse.load_npz(path_dash)
        metadata = np.load(path_meta_dash, allow_pickle=True)
        has_item_blocks = True
        print(">>> Matrice AVEC blocs item (vrai DASH)")
    else:
        X = sparse.load_npz(os.path.join(data_folder, f"history_features_{N_STUDENTS}std.npz"))
        metadata = np.load(os.path.join(data_folder, f"history_metadata_{N_STUDENTS}std.npz"),
                           allow_pickle=True)
        has_item_blocks = False
        print(">>> Matrice SANS blocs item (DASH approximé par CC)")

    n_users = len(metadata["user_ids"])
    n_items = len(metadata["item_ids"])
    n_kcs   = len(metadata["kc_list"])
    print(f"X.shape={X.shape}  n_users={n_users} n_items={n_items} n_kcs={n_kcs}")
    summ_user, _ = fit_models_cv(X, n_users, n_items, n_kcs, N_TW,
                                 split_mode="user", n_folds=N_FOLDS,
                                 seed=SEED, has_item_blocks=has_item_blocks)
    bayes_u = eval_bayes_cv(X, n_users, n_items, n_kcs, N_TW, n_folds=N_FOLDS, seed=SEED)
    us_u = eval_userskill_cv(data_folder, N_STUDENTS, N_TW,
                             split_mode="user", n_folds=N_FOLDS, seed=SEED)

    res_user = dict(summ_user)               # DAS3H, DASH, IRT/MIRT, PFA, AFM
    res_user["DAS3H+Bayes"] = bayes_u        # variante 1
    res_user["DAS3H user/skill"] = us_u      # variante 2 (None si matrice Alpha absente)
    summ_inter, _ = fit_models_cv(X, n_users, n_items, n_kcs, N_TW,
                                  split_mode="interaction", n_folds=N_FOLDS,
                                  seed=SEED, has_item_blocks=has_item_blocks)
    us_i = eval_userskill_cv(data_folder, N_STUDENTS, N_TW,
                             split_mode="interaction", n_folds=N_FOLDS, seed=SEED)

    res_inter = dict(summ_inter)
    res_inter["DAS3H+Bayes"] = None          # Bayes : pas de sens en interaction
    res_inter["DAS3H user/skill"] = us_i

    print_full_cv_table(res_user, res_inter, dataset=FOLDER, n_folds=N_FOLDS)

    

    print("\n!!! done !!!")