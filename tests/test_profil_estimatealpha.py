import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)


import src.Process.Estimator.AlphaEstimator as AE
import src.Process.DAS3H   as das3h
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from utils.this_queue import OurQueue
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
import random
NAME_FOLDER="Mathiadata" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 25351

def load_Model():
    # Load the model
    df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    kc_list=np.load(os.path.join(DATA_FOLDER,f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)["kc_list"]
    model_path = os.path.join(DATA_FOLDER, f"das3h_model_C0.1_{N_STUDENTS}std.pkl")
    X_full = sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_{N_STUDENTS}std.npz"))

    model, results = joblib.load(model_path)["model"], joblib.load(model_path)["results"]
    return df,q_matrix,kc_list,model,results,X_full



def CreateProfilMatrix(df, q_matrix, params):
    alpha_connus = params["alpha_s"]
    students = list(alpha_connus.keys())
    n_kc = q_matrix.shape[1]
    Q_profil = np.full((len(students), n_kc), np.nan)
    vector_alpha = np.empty(len(students))
 
    for i, s in enumerate(students):
        df_student = df[df["user_id"] == s].sort_values("timestamp")
        items = df_student["item_id"].values.astype(int)
        corrects = df_student["correct"].values
 
        n_correct_kc = np.zeros(n_kc)
        n_attempts_kc = np.zeros(n_kc)
 
        for item_id, correct in zip(items, corrects):
            kcs = np.where(q_matrix[item_id] == 1)[0]
            for kc in kcs:
                n_attempts_kc[kc] += 1
                if correct:
                    n_correct_kc[kc] += 1
 
        mask = n_attempts_kc > 0
        Q_profil[i, mask] = n_correct_kc[mask] / n_attempts_kc[mask]
        vector_alpha[i] = alpha_connus[s]
 
    return Q_profil, vector_alpha, students
 
    


 
def CreateProfilX(items_x, corrects_x, q_matrix, n_kc):
    profil = np.full(n_kc, np.nan)
    n_correct = np.zeros(n_kc)
    n_attempts = np.zeros(n_kc)
 
    for item_id, correct in zip(items_x, corrects_x):
        kcs = np.where(q_matrix[int(item_id)] == 1)[0]
        for kc in kcs:
            n_attempts[kc] += 1
            if correct:
                n_correct[kc] += 1
 
    mask = n_attempts > 0
    profil[mask] = n_correct[mask] / n_attempts[mask]
    return profil

#on pourra changer cette partie  pour une meilleur exploration 
def LookForKNN(profil_x, Q_profil, vector_alpha, K, min_kc_commun):
    kc_valid_x = np.where(~np.isnan(profil_x))[0]
    distances = []
 
    for i in range(Q_profil.shape[0]):
        kc_valid_s = np.where(~np.isnan(Q_profil[i]))[0]
        kc_communs = np.intersect1d(kc_valid_x, kc_valid_s)
 
        if len(kc_communs) < min_kc_commun:
            continue
 
        diff = profil_x[kc_communs] - Q_profil[i, kc_communs]
        dist = np.sqrt(np.mean(diff ** 2))
        distances.append((i, dist))
 
    distances.sort(key=lambda x: x[1])
    neighbours = distances[:K]
    return neighbours


def ComputePriorKNN(neighbours, vector_alpha, sigma_noyau):
    if not neighbours:
        return None, None
 
    poids = []
    alphas = []
 
    for idx, dist in neighbours:
        w = np.exp(-dist ** 2 / (2 * sigma_noyau ** 2))
        poids.append(w)
        alphas.append(vector_alpha[idx])
 
    poids = np.array(poids)
    alphas = np.array(alphas)
    somme_poids = np.sum(poids)
 
    if somme_poids < 1e-12:
        return None, None
 
    prior_mean_knn = np.sum(poids * alphas) / somme_poids
    variance = np.sum(poids * (alphas - prior_mean_knn) ** 2) / somme_poids
    prior_std_knn = np.sqrt(variance) if variance > 0 else 0.1
 
    return prior_mean_knn, prior_std_knn
 

def MelangerPriors(prior_mean_knn, prior_std_knn,prior_mean_global, prior_std_global,N, N0):
    lam = N / (N + N0)
    prior_mean = lam * prior_mean_knn + (1 - lam) * prior_mean_global
    prior_std = lam * prior_std_knn + (1 - lam) * prior_std_global
    return prior_mean, prior_std

def estimate_alpha_knn(X_student, y_student, items_x, corrects_x,
                       q_matrix, Q_profil, vector_alpha,
                       params, prior_mean_global, prior_std_global,
                       K=30, N0=15, sigma_noyau=0.3, min_kc_commun=3,
                       bounds=[-4,4]):
    n_kc = q_matrix.shape[1]
    N = len(y_student)
 
    profil_x = CreateProfilX(items_x, corrects_x, q_matrix, n_kc)
    neighbours = LookForKNN(profil_x, Q_profil, vector_alpha, K, min_kc_commun)
    
    print("Distances des 10 premiers voisins:", [round(d, 3) for _, d in neighbours[:10]])
    print("KC valides dans le profil:", np.sum(~np.isnan(profil_x)))

    if not neighbours:
        prior_mean = prior_mean_global
        prior_std = prior_std_global
    else:
        prior_mean_knn, prior_std_knn = ComputePriorKNN(neighbours, vector_alpha, sigma_noyau)
        if prior_mean_knn is None:
            prior_mean = prior_mean_global
            prior_std = prior_std_global
        else:
            prior_mean, prior_std =prior_mean_knn, prior_std_knn 
            """ MelangerPriors(
                prior_mean_knn, prior_std_knn,
                prior_mean_global, prior_std_global,
                N, N0
            )"""
 
    pipe = params["pipeline"]
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    coef = lr.coef_[0]
    intercept = lr.intercept_[0]
 
    X_scaled = scaler.transform(X_student)
    logits_base = X_scaled @ coef + intercept
 
    def neg_log_posterior(alpha_s):
        logits = logits_base + alpha_s
        p = 1 / (1 + np.exp(-logits))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = np.sum(y_student * np.log(p) + (1 - y_student) * np.log(1 - p))
        prior = -(alpha_s - prior_mean) ** 2 / (2 * prior_std ** 2)
        return -(ll + prior)
 
    result = minimize_scalar(neg_log_posterior, bounds=bounds, method='bounded')
    return result.x
 
def chooseStudent(df, params, min_interactions=1000, max_interactions=10000, n_test=50):
    
    interaction_counts = df.groupby("user_id").size()
    eligible = interaction_counts[
        (interaction_counts >= min_interactions) &
        (interaction_counts <= max_interactions)
    ].index.tolist()
    eligible = [s for s in eligible if s in params["alpha_s"]]
    students_test = eligible[:n_test]
    return eligible, students_test

def ComputePrior(params):
    
    alphas = list(params["alpha_s"].values())
    prior_mean = np.mean(alphas)
    prior_std =10
    bounds = [-4,4]
    return prior_mean, prior_std, bounds

def splitX_train(X_student, N, n_users, offset):
    y_all = X_student[:, 3].toarray().flatten()
    cols = list(range(X_student.shape[1]))
    cols.remove(3)
    X_features = X_student[:, cols]
    X_dense = X_features.toarray()
    X_dense[:, offset:offset + n_users] = 0
    X_train, X_test = X_dense[:N], X_dense[N:]
    y_train, y_test = y_all[:N], y_all[N:]
    return (sparse.csr_matrix(X_train), y_train), (sparse.csr_matrix(X_test), y_test)
 

def extract_items_corrects(X_student, N):
    items = X_student[:N, 1].toarray().flatten().astype(int)
    corrects = X_student[:N, 3].toarray().flatten()
    return items, corrects

def validate_split_temporel(model, X_full, df, q_matrix, students_test,
                            estimator, Q_profil, vector_alpha,
                            prior_mean_global, prior_std_global,
                            n_values=None, min_test=30, min_pos_neg=3,
                            K=30, N0=15, sigma_noyau=0.3, min_kc_commun=3,params=None):
    if n_values is None:
        n_values = [10, 20, 30, 50, 75, 100, 150, 200, 1000]
 
    
    
    pipe = model.model
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    coef = lr.coef_[0]
    intercept = lr.intercept_[0]
    offset = 4
    n_users = model.n_users
    all_user_ids = X_full[:, 0].toarray().flatten()
 
    results = []
    all_y, all_p_zero, all_p_base, all_p_knn = [], [], [], []
 
    for idx, s in enumerate(students_test):
        print(f"Step: {idx}/{len(students_test)}")
        alpha_vrai = params["alpha_s"].get(s, None)
        if alpha_vrai is None: # pourquoi ici on skip parce qu'on cherche à comparer avec le vrai sion dans la vrai vie on va regarder ce qui n'est exploré par le modèle 
            continue
 
        mask = all_user_ids == s
        X_student = X_full[mask]
        n_total = X_student.shape[0]
 
        for N in n_values:
            if n_total - N < min_test:
                continue
 
            (X_train, y_train), (X_test, y_test) = splitX_train(X_student, N, n_users, offset)
            #assez de succés et d'échec pour calculer l'AUC 
            if y_test.sum() < min_pos_neg or (len(y_test) - y_test.sum()) < min_pos_neg:
                continue
            items_x, corrects_x = extract_items_corrects(X_student, N)
            alpha_base = estimator.estimate_from_X(X_train, y_train)
            alpha_knn = estimate_alpha_knn(
                X_train, y_train, items_x, corrects_x,
                q_matrix, Q_profil, vector_alpha,
                params, prior_mean_global, prior_std_global,
                K=K, N0=N0, sigma_noyau=sigma_noyau,
                min_kc_commun=min_kc_commun
            )
            X_test_scaled = scaler.transform(X_test)
            logits_base = X_test_scaled @ coef + intercept
            p_zero = 1 / (1 + np.exp(-logits_base))
            p_base = 1 / (1 + np.exp(-(logits_base + alpha_base)))
            p_knn = 1 / (1 + np.exp(-(logits_base + alpha_knn)))
            p_vrai = 1 / (1 + np.exp(-(logits_base + alpha_vrai)))
            all_y.append(y_test)
            all_p_zero.append(p_zero)
            all_p_base.append(p_base)
            all_p_knn.append(p_knn)
            results.append({
                "student": s,
                "N_train": N,
                "N_test": n_total - N,
                "alpha_vrai": alpha_vrai,
                "alpha_base": alpha_base,
                "alpha_knn": alpha_knn,
                "erreur_base": abs(alpha_base - alpha_vrai),
                "erreur_knn": abs(alpha_knn - alpha_vrai),
                "logloss_zero": log_loss(y_test, p_zero),
                "logloss_base": log_loss(y_test, p_base),
                "logloss_knn": log_loss(y_test, p_knn),
                "logloss_vrai": log_loss(y_test, p_vrai),
            })
 
    all_y = np.concatenate(all_y)
    all_p_zero = np.concatenate(all_p_zero)
    all_p_base = np.concatenate(all_p_base)
    all_p_knn = np.concatenate(all_p_knn)
 
    df_results = pd.DataFrame(results)
    df_summary = df_results.groupby("N_train").agg(
        n_students=("student", "count"),
        logloss_zero_mean=("logloss_zero", "mean"),
        logloss_base_mean=("logloss_base", "mean"),
        logloss_knn_mean=("logloss_knn", "mean"),
        logloss_vrai_mean=("logloss_vrai", "mean"),
        erreur_base_mean=("erreur_base", "mean"),
        erreur_knn_mean=("erreur_knn", "mean"),
        erreur_base_std=("erreur_base", "std"),
        erreur_knn_std=("erreur_knn", "std"),
    ).reset_index()
 
    return all_y, all_p_zero, all_p_base, all_p_knn, df_summary, df_results

def plot_erreur_alpha_compare(df_summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    N = df_summary["N_train"]
    ax.plot(N, df_summary["erreur_base_mean"], "D-", color="red", label="MAP prior global")
    ax.plot(N, df_summary["erreur_knn_mean"], "s-", color="blue", label="MAP prior kNN")
    ax.set_xlabel("N (interactions train)")
    ax.set_ylabel("|α estimé − α vrai|")
    ax.set_title("Convergence : prior global vs prior kNN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_alpha_convergence_compare(df_results, students_plot):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for s in students_plot:
        df_s = df_results[df_results["student"] == s]
        if df_s.empty:
            continue
        N = df_s["N_train"]
        alpha_vrai = df_s["alpha_vrai"].iloc[0]
 
        line, = axes[0].plot(N, df_s["alpha_base"], marker='o', label=f"Élève {s}")
        axes[0].axhline(y=alpha_vrai, color=line.get_color(), linestyle='--', alpha=0.4)
 
        line2, = axes[1].plot(N, df_s["alpha_knn"], marker='o', label=f"Élève {s}")
        axes[1].axhline(y=alpha_vrai, color=line2.get_color(), linestyle='--', alpha=0.4)
 
    axes[0].set_title("MAP prior global")
    axes[1].set_title("MAP prior kNN")
    for ax in axes:
        ax.set_xlabel("N (interactions train)")
        ax.set_ylabel("α estimé")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df, q_matrix, kc_list, model, results, X_full = load_Model()
    params = model.get_params()
    eligible, students_test = chooseStudent( df,params=params, min_interactions=500, n_test=20)
 
    
    params["pipeline"] = model.model
    prior_mean, prior_std, bounds = ComputePrior(params=params)
 
    estimator = AE.alphaEstimator(
        params=params, prior_mean=prior_mean,
        prior_std=prior_std, bounds=bounds
    )
 
    print("Construction de la matrice de profils...")
    Q_profil, vector_alpha, students_ref = CreateProfilMatrix(df, q_matrix, params)
    print(f"Matrice de profils : {Q_profil.shape}")
 
    n_values = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500]
 
    all_y, all_p_zero, all_p_base, all_p_knn, df_sum, df_res = validate_split_temporel(
        model, X_full, df, q_matrix, students_test,
        estimator, Q_profil, vector_alpha,
        prior_mean, prior_std,
        n_values=n_values, min_test=30, min_pos_neg=3,
        K=50, N0=10, sigma_noyau=0.5, min_kc_commun=10,params=params
    )
    plot_erreur_alpha_compare(df_sum)
    plot_alpha_convergence_compare(df_res, random.sample(list(students_test), 5))
    print("Done!")

print("!!!!!!!!!!!!!!!!!!!!stop!!!!!!!!!!!!!!!!!!!!")