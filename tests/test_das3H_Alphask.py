import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
import pandas as pd
from scipy import sparse

import src.datamodel.Studentdata as SD
import matplotlib.pyplot as plt
import src.Process.DAS3H as DAS3H
import src.datamodel.Historydata as HIS
import joblib
from utils.this_queue import OurQueue
from collections import defaultdict
import time
from sklearn.calibration import calibration_curve

from sklearn.metrics import brier_score_loss, log_loss
NAME_FOLDER="bridge_algebra06" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 1146# Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30
MODEL_C = 0.01  # Regularization parameter
N_TIME_WINDOWS = 5



def prepare_featuresAlpha( data_folder,df,q_matrix: np.ndarray, stdmodel: SD.StudentDATA ):
    print("!!!!!!!!!!!!!!!!Preparing history !!!!!!!!!!!!")
    his = HIS.HistoryDATA(stdmodel=stdmodel)
    X, user_ids, item_ids, listKC = his.ComputeHistoryFeaturesALPHASK(Q_mat=q_matrix, df=df)
    #save X to npz file
    sparse.save_npz(os.path.join(data_folder, f"history_features_Alpha{N_STUDENTS}std.npz"), sparse.csr_matrix(X))
    np.savez(os.path.join(data_folder, f"history_metadata_Alpha{N_STUDENTS}std.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listKC)
    return X, user_ids, item_ids, listKC


def test_model_training(data_folder,X, user_ids, item_ids, kc_list, model_c: list[float] = [0.01, 0.1, 1.0], n_tw: int = 5,perc_init: float = 0.2):
    modeldict = {}
    for c in model_c:
        print("Das3h    with C =", c)
        model = DAS3H.DAS3HModel(C=c)
        results = model.fit(
        X,
        user_ids=user_ids,
        item_ids=item_ids,
        kc_list=kc_list,
        n_tw=n_tw,
        perc_init=perc_init)
        modeldict[c] = (model, results)
        print(f" AUC:  {results['AUC']:.4f}")
        print(f" NLL:  {results['NLL']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
    #save modeldict to npz file
    for c, (model, results) in modeldict.items():
        joblib.dump(
            {"model": model, "results": results},
            os.path.join(data_folder, f"das3h_model_Alpha_C{c}_{N_STUDENTS}std.pkl")
        )
    return modeldict



def test_model_parameters(model: DAS3H.DAS3HModel):
    params = model.get_params_AlphaRatio()
    #Visualiser l'ability des élèves avec la variance et l'ecart type 
    print("Number of students:", len(params["alpha_s"]))
    print("Mean ability (alpha_s):", np.mean(list(params["alpha_s"].values())))
    print("Variance of ability (alpha_s):", np.var(list(params["alpha_s"].values())))
    print("Std ability (alpha_s):", np.std(list(params["alpha_s"].values())))
    #Visualiser les diffcultes des élèves avec la variance et l'ecart type 
    print("Number of items:", len(params["delta_j"]))
    print("Mean difficulty (delta_j):", np.mean(list(params["delta_j"].values())))
    print("Variance of difficulty (delta_j):", np.var(list(params["delta_j"].values())))
    print("Std difficulty (delta_j):", np.std(list(params["delta_j"].values())))
    #Visualiser les facilités des KC avec la variance et l'ecart type
    print("Number of KCs:", len(params["beta_k"]))
    print("Mean ease (beta_k):", np.mean(list(params["beta_k"].values())))
    print("Variance of ease (beta_k):", np.var(list(params["beta_k"].values())))
    print("Std ease (beta_k):", np.std(list(params["beta_k"].values())))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alpha_values = list(params["alpha_sk"].values())
    axes[0].hist(alpha_values, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(np.mean(alpha_values), color="red", linestyle="--", label="mean")
    axes[0].set_title(f"Capacité élèves (alpha_s)\n"
                      f"mean={np.mean(alpha_values):.3f}, "
                      f"std={np.std(alpha_values):.3f}")
    axes[0].set_xlabel("alpha_s")
    axes[0].legend()

    # --- Delta_j : distribution des difficultés items ---
    delta_values = list(params["delta_j"].values())
    axes[1].hist(delta_values, bins=30, color="salmon", edgecolor="white")
    axes[1].axvline(np.mean(delta_values), color="red", linestyle="--", label="mean")
    axes[1].set_title(f"Difficulté items (delta_j)\n"
                      f"mean={np.mean(delta_values):.3f}, "
                      f"std={np.std(delta_values):.3f}")
    axes[1].set_xlabel("delta_j")
    axes[1].legend()

    # --- Beta_k : distribution des facilités KCs ---
    beta_values = list(params["beta_k"].values())
    axes[2].hist(beta_values, bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].axvline(np.mean(beta_values), color="red", linestyle="--", label="mean")
    axes[2].set_title(f"Facilité KCs (beta_k)\n"
                      f"mean={np.mean(beta_values):.3f}, "
                      f"std={np.std(beta_values):.3f}")
    axes[2].set_xlabel("beta_k")
    axes[2].legend()
    plt.tight_layout()
    plt.show()
    return params



def plot_calibration1(y_true,y_trueg, y_pred_alpha,ypred_global=None):
    #plot la courbe de calibration du modèle original
    fig, ax = plt.subplots(figsize=(7, 7))
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_alpha, n_bins=10
    )   
    prob_true_g, prob_pred_global= calibration_curve(
        y_trueg, ypred_global, n_bins=10
    )  
    ax.plot(prob_pred, prob_true,
            marker='o', color="steelblue",
            linewidth=2, label="modèle user_kcs")
    
    ax.plot(prob_pred_global, prob_true_g,
            marker='o', color="salmon",
            linewidth=2, label="modèle normal")
    ax.plot([0, 1], [0, 1],
            linestyle="--", color="gray",
            label="calibration parfaite")
    ax.fill_between(prob_pred,
                    prob_true - 0.05,
                    prob_true + 0.05,
                    alpha=0.1, color="steelblue",
                    label="marge ±5%")
    
    ax.fill_between(prob_pred_global,
                    prob_true - 0.05,
                    prob_true + 0.05,
                    alpha=0.1, color="salmon",
                    label="marge ±5%")
    ax.set_xlabel("P prédite")
    ax.set_ylabel("Taux réel de réussite")
    ax.legend()
    ax.grid(True, alpha=0.3)    


    plt.suptitle(f"Calibration — P prédite vs taux réel\n"
                f"Courbe proche diagonale = bien calibré \n"
                 f"Dataset : {NAME_FOLDER}",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_calibration_all_datasets_alpha():
    """Trace les courbes de calibration (variante Alpha = user/kc) des 4 jeux
    de données sur une seule et même courbe."""

    # (nom_dossier, n_students, titre affiché, couleur)
    datasets = [
        ("Mathiadata",        25351, "MATHIA",            "steelblue"),
        ("ASSISTments13_12",  15698, "ASSISTments",       "salmon"),
        ("bridge_algebra06",  1146,  "Bridge Algebra 06", "mediumseagreen"),
        ("algebra05",         574,   "Algebra 05",        "darkorange"),
    ]

    model_c_list = [0.01, 0.1, 1]

    fig, ax = plt.subplots(figsize=(9, 9))

    for folder, n_students, title, color in datasets:
        data_folder = os.path.join("data", folder)

        # --- Charger le meilleur modèle Alpha selon le score AUC - NLL - RMSE ---
        best_score = -float("inf")
        best_model = None
        best_results = None
        best_c = None

        for c in model_c_list:
            model_path = os.path.join(
                data_folder, f"das3h_model_Alpha_C{c}_{n_students}std.pkl"
            )
            if not os.path.exists(model_path):
                continue
            loaded = joblib.load(model_path)
            results = loaded["results"]
            score = results["AUC"] - results["NLL"] - results["RMSE"]
            if score > best_score:
                best_score = score
                best_model = loaded["model"]
                best_results = results
                best_c = c

        if best_model is None:
            print(f"[!] Aucun modèle Alpha trouvé pour {title}")
            continue

        # --- Charger les features Alpha et calculer les prédictions ---
        X = sparse.load_npz(
            os.path.join(data_folder, f"history_features_Alpha{n_students}std.npz")
        )
        cols = list(range(X.shape[1]))
        cols.remove(3)
        X_no_col3 = X[:, cols]

        p_pred_raw = best_model.predict_proba(X_no_col3)
        p_pred = p_pred_raw if p_pred_raw.ndim == 1 else p_pred_raw[:, 1]
        y_true = X[:, 3].toarray().flatten()

        # --- Courbe de calibration ---
        prob_true, prob_pred = calibration_curve(y_true, p_pred, n_bins=10)

        ax.plot(prob_pred, prob_true, marker="o", color=color, linewidth=2,
                label=f"{title} (C={best_c}, AUC={best_results['AUC']:.3f})")

    # --- Référence : calibration parfaite ---
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="calibration parfaite")
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                    alpha=0.08, color="green", label="marge ±5%")

    ax.set_xlabel("P prédite DAS3H_V2 ",fontsize=15)
    ax.set_ylabel("Taux réel de réussite",fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Calibration (variante user/kc) — P prédite vs taux réel\n"
                 "Courbe proche de la diagonale = bien calibré (4 jeux de données)",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    timetoexeucte=4#Time to execute

    if timetoexeucte==1: 
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        start=time.time()
        X_alpha, user_ids, item_ids, kc_list = prepare_featuresAlpha(DATA_FOLDER,df,q_matrix, stdmodel=None)
        end=time.time()
        print(f"Time to prepare features: {end - start:.2f} seconds")
    elif timetoexeucte==2:

        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        X_alpha=sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_Alpha{N_STUDENTS}std.npz"))
        metadata = np.load(os.path.join(DATA_FOLDER, f"history_metadata_Alpha{N_STUDENTS}std.npz"), allow_pickle=True)
        user_ids = metadata["user_ids"]
        item_ids = metadata["item_ids"]
        kc_list = metadata["kc_list"]
        start=time.time()
        modeldict = test_model_training(DATA_FOLDER,X_alpha, user_ids, item_ids, kc_list, model_c=[0.01,0.1,1], n_tw=N_TIME_WINDOWS, perc_init=0.8)
        end=time.time()
        print(f"Time to train model: {end - start:.2f} seconds")

    elif timetoexeucte==3:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))  
        X_alpha=sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_Alpha{N_STUDENTS}std.npz"))
        X_global=sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_{N_STUDENTS}std.npz"))
        model_pathR = os.path.join(DATA_FOLDER, f"das3h_model_Alpha_C1_{N_STUDENTS}std.pkl")
        model_Alpha, results_Alpha = joblib.load(model_pathR)["model"], joblib.load(model_pathR)["results"]
        model_pathoriginal = os.path.join(DATA_FOLDER, f"das3h_model_C1_{N_STUDENTS}std.pkl")
        model_orginal, results_original = joblib.load(model_pathoriginal)["model"], joblib.load(model_pathoriginal)["results"]
        
        cols = list(range(X_alpha.shape[1]))
        cols.remove(3)
        X_no_col3 = X_alpha[:, cols]
        p_pred_raw = model_Alpha.predict_proba(X_no_col3)
        p_pred = p_pred_raw if p_pred_raw.ndim == 1 else p_pred_raw[:, 1]

        cols_g= list(range(X_global.shape[1]))
        cols_g.remove(3)
        X_no_col3_g = X_global[:, cols_g]
        p_pred_or = model_orginal.predict_proba(X_no_col3_g)
        p_pred_or= p_pred_or if p_pred_or.ndim == 1 else p_pred_or[:, 1]
        y_true = X_alpha[:, 3].toarray().flatten()
        y_trueg = X_global[:, 3].toarray().flatten()

        plot_calibration1(y_true,y_trueg, p_pred,ypred_global=p_pred_or)

    elif timetoexeucte == 4:
        plot_calibration_all_datasets_alpha()
        print("Courbes de calibration Alpha des 4 datasets tracées")
        
        
    print("Done")
    

    print("!!!!!!!!!!!!Done!!!!!!!!!!!!!!!!!!!")