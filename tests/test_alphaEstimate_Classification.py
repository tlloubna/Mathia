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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def chooseStudent(model, df, min_interactions=1000, max_interactions=5000, n_test=50):
    params = model.get_params()
    interaction_counts = df.groupby("user_id").size()
    eligible = interaction_counts[
        (interaction_counts >= min_interactions) & 
        (interaction_counts <= max_interactions)
    ].index.tolist()
    eligible = [s for s in eligible if s in params["alpha_s"]]
    students_test = eligible[:n_test]
    return eligible, students_test


def splitX_train(X_student,N,n_users,offset):
    y_all = X_student[:, 3].toarray().flatten()
    cols = list(range(X_student.shape[1]))
    cols.remove(3)
    X_features = X_student[:, cols]
    X_dense = X_features.toarray()
    X_dense[:, offset:offset + n_users] = 0
    X_train,X_test=X_dense[:N],X_dense[N:]
    y_train,y_test=y_all[:N],y_all[N:]
    return (sparse.csr_matrix(X_train), y_train), (sparse.csr_matrix(X_test), y_test)

def validate_split_temporel(model, X_full, students_test, estimator,
                            n_values=[10, 20, 30, 50, 75, 100, 150, 200,1000],
                            min_test=30, min_pos_neg=3):
    params = model.get_params()
    pipe = model.model
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    coef = lr.coef_[0]
    intercept = lr.intercept_[0]
    offset = 4
    n_users = model.n_users
    all_user_ids = X_full[:, 0].toarray().flatten()
    results = []
    all_y = []
    all_p_zero = []
    all_p_estime = []
    all_p_vrai = []
    student_masks = {}
    
    for s in students_test:
        student_masks[s] = np.where(all_user_ids == s)[0]
    for idx,s in enumerate(students_test):
        print("Step :",idx,"/",len(students_test))
        alpha_vrai = params["alpha_s"].get(s, None)
        if alpha_vrai is None:
            continue
        #mask = all_user_ids == s
        #X_student = X_full[mask]
        X_student = X_full[student_masks[s]]
        n_total = X_student.shape[0]
        for N in n_values:
            if n_total - N < min_test:
                continue
            (X_train, y_train), (X_test, y_test) = splitX_train(X_student, N, n_users, offset)
            if y_test.sum() < min_pos_neg or (len(y_test) - y_test.sum()) < min_pos_neg:
                continue
            alpha_estime ,alpha_std= estimator._get_prior_from_cluster(X_train, y_train)
            X_test_scaled = scaler.transform(X_test)
            logits_base = X_test_scaled @ coef + intercept
            p_zero = 1 / (1 + np.exp(-logits_base))
            p_estime = 1 / (1 + np.exp(-(logits_base + alpha_estime)))
            p_vrai = 1 / (1 + np.exp(-(logits_base + alpha_vrai)))
            all_y.append(y_test)
            all_p_zero.append(p_zero)
            all_p_estime.append(p_estime)
            all_p_vrai.append(p_vrai)
            results.append({
                "student": s,
                "N_train": N,
                "N_test": n_total - N,
                "alpha_vrai": alpha_vrai,
                "alpha_estime": alpha_estime,
                "erreur_alpha": abs(alpha_estime - alpha_vrai),
                "logloss_zero": log_loss(y_test, p_zero), 
            "logloss_estime": log_loss(y_test, p_estime),
            "logloss_vrai": log_loss(y_test, p_vrai),
            })
    all_y = np.concatenate(all_y)
    all_p_zero = np.concatenate(all_p_zero)
    all_p_estime = np.concatenate(all_p_estime)
    all_p_vrai = np.concatenate(all_p_vrai)
    df_results = pd.DataFrame(results)
    df_summary = df_results.groupby("N_train").agg(
    n_students=("student", "count"),
    logloss_zero_mean=("logloss_zero", "mean"),
    logloss_estime_mean=("logloss_estime", "mean"),
    logloss_vrai_mean=("logloss_vrai", "mean"),
    erreur_alpha_mean=("erreur_alpha", "mean"),
    erreur_alpha_std=("erreur_alpha", "std"),).reset_index()
    L_zero = df_summary["logloss_zero_mean"]
    L_vrai = df_summary["logloss_vrai_mean"]
    L_estime = df_summary["logloss_estime_mean"]
    df_summary["ratio_gap"] = (L_zero - L_estime) / (L_zero - L_vrai)
    return all_y,all_p_zero,all_p_estime,all_p_vrai,df_summary, df_results

def CreateFeatureAlpha(params, df):
    student_ids = list(params["alpha_s"].keys())
    student_stats = df.groupby("user_id").agg(
        mean_success_rate=("correct", "mean"),
    )
    df["delta_j"] = df["item_id"].map(params["delta_j"])
    mean_deltas = df.groupby("user_id")["delta_j"].mean()

    features, alphas, valid_students = [], [], []
    for s in student_ids:
        if s not in student_stats.index:
            continue
        stats = student_stats.loc[s]
        features.append([stats["mean_success_rate"], mean_deltas.get(s, 0)])
        alphas.append(params["alpha_s"][s])   # gardé à part, PAS dans le clustering
        valid_students.append(s)

    return np.array(features), np.array(alphas), valid_students

def FitKmeans(X,K_max):
    X_scaled=StandardScaler().fit_transform(X)
    Clusters={}
    for k in range(2,K_max):
        kmeans=KMeans(n_clusters=k,random_state=42).fit(X_scaled)
        Clusters[k]=kmeans

    return Clusters,X_scaled

def plot_clusters_pca( labels, X_scaled, feature_names=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 7))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    label=f"Cluster {label}", alpha=0.6, s=30)
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Clusters d'élèves (ACP)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if feature_names is None:
        feature_names = ["alpha_s", "mean_success_rate", 
                         "n_interactions", "n_kcs_seen", "mean_delta"]
    
    print("\nContribution des features aux composantes :")
    for i, name in enumerate(feature_names):
        print(f"  {name:25s}  PC1={pca.components_[0][i]:+.3f}  "
              f"PC2={pca.components_[1][i]:+.3f}")

def remove_outliers(X, alphas, valid_students, percentile=99):
    X_df = pd.DataFrame(X, columns=["mean_success_rate", "mean_delta"])
    mask = np.ones(len(X), dtype=bool)
    for col in ["mean_success_rate", "mean_delta"]:
        upper = np.percentile(X_df[col], percentile)
        lower = np.percentile(X_df[col], 100 - percentile)
        mask &= (X_df[col] <= upper) & (X_df[col] >= lower)
    X_clean = X[mask]
    alphas_clean = alphas[mask]
    students_clean = [s for s, m in zip(valid_students, mask) if m]
    print(f"Retirés : {len(X) - len(X_clean)} outliers")
    return X_clean, alphas_clean, students_clean

def plotalphaEstime(df_results):
    students = df_results["student"].unique()  
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in students:
        df_s = df_results[df_results["student"] == s]  
        N = df_s["N_train"]
        alpha_estime = df_s["alpha_estime"]  
        alpha_vrai = df_s["alpha_vrai"].iloc[0]
        line, = ax.plot(N, alpha_estime, marker='o', label=f"Élève {s}")
        ax.axhline(y=alpha_vrai, color=line.get_color(), linestyle='--', alpha=0.4)
    ax.set_xlabel("N (interactions train)")  
    ax.set_ylabel("Alpha estimé")
    ax.set_title("Évolution de α estimé en fonction de N")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    df, q_matrix, kc_list, model, results, X_full = load_Model()
    params = model.get_params()
    params["pipeline"] = model.model                        
    
    X_Fkmeans,alphas, valides_student = CreateFeatureAlpha(params=params, df=df)
    X_clean, alphas_clean, students_clean = remove_outliers(X_Fkmeans, alphas, valides_student, percentile=99)
    Clusters, X_scaled = FitKmeans(X_clean, K_max=30)
    scaler_kmeans = StandardScaler().fit(X_clean)  
    """for k in range(2,10):
        Labels = Clusters[k].labels_
        plot_clusters_pca( Labels, X_scaled, feature_names=[ "alpha_s","mean_success_rate",  "mean_delta"])"""
    Labels = Clusters[4].labels_
    cluster_stats = {}
    for k in range(4):
        mask = Labels == k
        alphas_cluster = alphas_clean[mask]   # vrais alpha, pas X_clean[:,0]
        cluster_stats[k] = {
            "mean_alpha": np.mean(alphas_cluster),
            "std_alpha": np.std(alphas_cluster),
        }
    

    estimator_cluster = AE.AlphaEstimatorUsingClass(
        params=params, 
        kmeans=Clusters[4],
        scaler_kmeans=scaler_kmeans,
        cluster_stats=cluster_stats,
    )
    eligible, students_test = chooseStudent(model, df, 
                                            min_interactions=500,
                                            max_interactions=1000, 
                                            n_test=20)
    
    print("=== Prior CLUSTER ===")
    _, _, _, _, df_sum_cluster, df_res_cluster = validate_split_temporel(
        model, X_full, students_test, estimator_cluster,
        n_values=[1,3,5,10, 20, 30, 40,50, 60,70,80, 90,100, 150, 200,300,400, 500])
    plotalphaEstime(df_res_cluster)
    
    
    
    print("!!!!!!!!!!Done!!!!")
print("!!!!!!!!!Done!!!!!!!!!!!!!!!")