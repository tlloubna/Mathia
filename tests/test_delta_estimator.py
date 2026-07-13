import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)


import src.Process.Estimator.DeltaEstimator as DE
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss
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

def chooseItems(model, df, min_interactions=1000, max_interactions=5000, n_test=50):
    params = model.get_params()
    interaction_counts = df.groupby("item_id").size()
    eligible = interaction_counts[
        (interaction_counts >= min_interactions) & 
        (interaction_counts <= max_interactions)
    ].index.tolist()
    eligible = [s for s in eligible if s in params["delta_j"]]
    item_test = eligible[:n_test]
    return eligible, item_test

def ComputePrior(model):
    params = model.get_params()
    deltas = list(params["delta_j"].values())
    prior_mean = np.mean(deltas)
    prior_std =np.std(deltas)
    bounds = [min(deltas), max(deltas)]
    return prior_mean, prior_std, bounds

def splitX_train(X_student, N, n_users, n_items, offset_users=4):
    y_all = X_student[:, 3].toarray().flatten()
    cols = list(range(X_student.shape[1]))
    cols.remove(3)
    X_features = X_student[:, cols]
    X_dense = X_features.toarray()

    offset_items = offset_users + n_users          
    X_dense[:, offset_items:offset_items + n_items] = 0  

    X_train, X_test = X_dense[:N], X_dense[N:]
    y_train, y_test = y_all[:N], y_all[N:]
    return (sparse.csr_matrix(X_train), y_train), (sparse.csr_matrix(X_test), y_test)

def validate_split_temporel(model, X_full, item_test, estimator,
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
    n_items=model.n_items
    all_item_ids = X_full[:,1].toarray().flatten()
    results = []
    all_y = []
    all_p_zero = []
    all_p_estime = []
    all_p_vrai = []
    items_masks = {}
    
    for s in item_test:
        items_masks[s] = np.where(all_item_ids == s)[0]
    for idx,s in enumerate(item_test):
        print("Step :",idx,"/",len(item_test))
        delta_vrai = params["delta_j"].get(s, None)
        if delta_vrai is None:
            continue

        
        #mask = all_user_ids == s
        #X_student = X_full[mask]
        X_items = X_full[items_masks[s]]
        n_total = X_items.shape[0]
        for N in n_values:
            
            if n_total - N < min_test:
                continue
            (X_train, y_train), (X_test, y_test) = splitX_train(X_items, N, n_users,n_items, offset)
            if y_test.sum() < min_pos_neg or (len(y_test) - y_test.sum()) < min_pos_neg:
                continue
            delta_estime = estimator.estimate_from_X(X_train, y_train)
            X_test_scaled = scaler.transform(X_test)
            logits_base = X_test_scaled @ coef + intercept
            p_zero = 1 / (1 + np.exp(-logits_base))
            p_estime = 1 / (1 + np.exp(-(logits_base +delta_estime)))
            p_vrai = 1 / (1 + np.exp(-(logits_base - delta_vrai)))
            all_y.append(y_test)
            all_p_zero.append(p_zero)
            all_p_estime.append(p_estime)
            all_p_vrai.append(p_vrai)
            results.append({
                "item": s,
                "N_train": N,
                "N_test": n_total - N,
                "delta_vrai": delta_vrai,
                "delta_estime": -delta_estime,
                "erreur_delta": abs(-delta_estime - delta_vrai),
                "logloss_zero": log_loss(y_test, p_zero), #L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))
            "logloss_estime": log_loss(y_test, p_estime),
            "logloss_vrai": log_loss(y_test, p_vrai),
            })
            
    all_y = np.concatenate(all_y)
    all_p_zero = np.concatenate(all_p_zero)
    all_p_estime = np.concatenate(all_p_estime)
    all_p_vrai = np.concatenate(all_p_vrai)
    df_results = pd.DataFrame(results)
    df_summary = df_results.groupby("N_train").agg(
    n_students=("item", "count"),
    logloss_zero_mean=("logloss_zero", "mean"),
    logloss_estime_mean=("logloss_estime", "mean"),
    logloss_vrai_mean=("logloss_vrai", "mean"),
    erreur_delta_mean=("erreur_delta", "mean"),
    erreur_delta_std=("erreur_delta", "std"),).reset_index()
    L_zero = df_summary["logloss_zero_mean"]
    L_vrai = df_summary["logloss_vrai_mean"]
    L_estime = df_summary["logloss_estime_mean"]
    df_summary["ratio_gap"] = (L_zero - L_estime) / (L_zero - L_vrai)
    
    AUC_b=   roc_auc_score(all_y,all_p_estime)
    NLL_b=  log_loss(all_y,all_p_estime)
    RMSE_b=   np.sqrt(mean_squared_error(all_y,all_p_estime))

    AUC_v=   roc_auc_score(all_y,all_p_vrai)
    NLL_v=  log_loss(all_y,all_p_vrai)
    RMSE_v=   np.sqrt(mean_squared_error(all_y,all_p_vrai))

    AUC_z=   roc_auc_score(all_y,all_p_zero)
    NLL_z=log_loss(all_y,all_p_zero)
    RMSE_z=   np.sqrt(mean_squared_error(all_y,all_p_zero))

    print("======Results Bayes==========")
    print("AUC Bayes:",AUC_b)
    print("NLL Bayes :", NLL_b)
    print("RMSE Bayes :",RMSE_b)

    print("======Results Das3h==========")
    print("AUC da3h:",AUC_v)
    print("NLL das3H :", NLL_v)
    print("RMSE da3h:",RMSE_v)

    print("======Results Zero==========")
    print("AUC Zero:",AUC_z)
    print("NLL Zero :", NLL_z)
    print("RMSE Zero :",RMSE_z)
    return all_y,all_p_zero,all_p_estime,all_p_vrai,df_summary, df_results



def plot_calibration_compare(all_y, all_p_zero, all_p_estime, all_p_vrai):
    fig, ax = plt.subplots(figsize=(10, 6))

    for p, label, color in [
        (all_p_zero, "delta = 0", "gray"),
        (all_p_estime, "delta estimé", "blue"),
        (all_p_vrai, "delta vrai", "green"),
    ]:
        prob_true, prob_pred = calibration_curve(all_y, p, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', color=color,
                linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="calibration parfaite")
    ax.set_xlabel("P prédite")
    ax.set_ylabel("Taux réel de réussite")
    ax.set_title("Calibration — Split temporel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_logloss(df_summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    N = df_summary["N_train"]
    ax.plot(N, df_summary["logloss_zero_mean"], "o--", color="gray", label="delta = 0")
    ax.plot(N, df_summary["logloss_estime_mean"], "s-", color="blue", label="delta estimé")
    ax.plot(N, df_summary["logloss_vrai_mean"], "^-", color="green", label="delta vrai")
    ax.set_xlabel("N (interactions train)")
    ax.set_ylabel("Log-loss sur bloc test")
    ax.set_title("Log-loss — Split temporel")
    ax.legend()
    ax.grid(True, alpha=0.3)    
    plt.tight_layout()
    plt.show()


def plotdeltaEstime(df_results):
    items = df_results["item"].unique()  
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in items:
        df_s = df_results[df_results["item"] == s]  
        N = df_s["N_train"]
        delta_estime = df_s["delta_estime"]  
        delta_vrai = df_s["delta_vrai"].iloc[0]
        line, = ax.plot(N, delta_estime, marker='o', label=f"Item {s}")
        ax.axhline(y=delta_vrai, color=line.get_color(), linestyle='--', alpha=0.4)
    ax.set_xlabel("N (interactions train)")  
    ax.set_ylabel("delta estimé")
    ax.set_title("Évolution de delta estimé en fonction de N")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    df, q_matrix, kc_list, model, results, X_full = load_Model()
    params = model.get_params()
    params["pipeline"] = model.model                        

    
    prior_mean, prior_std, bounds = ComputePrior(model=model)
    estimator_global = DE.DeltaEstimator(
        params=params, prior_mean=prior_mean,
        prior_std=prior_std, bounds=[-3, 3],
    )
    
    
    eligible, item_test=chooseItems(model, df, min_interactions=1000, max_interactions=5000, n_test=10)
    all_y,all_p_zero,all_p_estime,all_p_vrai,df_sum_global, df_res_global = validate_split_temporel(
        model, X_full,item_test, estimator_global,
        n_values=[1,3,5,10, 20, 30, 40,50, 60,70,80, 90,100, 150, 200,300,400, 500])
    
    plot_calibration_compare(all_y, all_p_zero, all_p_estime, all_p_vrai)
    plotdeltaEstime(df_res_global)
    plot_logloss(df_sum_global)
    #plot_compare(df_sum_global, df_sum_cluster)
    
    print("!!!!!!!!!!Done!!!!")
print("!!!!!!!!!Done!!!!!!!!!!!!!!!")


            

                
