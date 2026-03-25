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
import src.graphics.PlotOutills as Plot
import src.graphics.das3hviz as Vis
import joblib
import time
import seaborn as sns
from utils.this_queue import OurQueue
from collections import defaultdict
from sklearn.calibration import calibration_curve

NAME_FOLDER="Mathiadata"
DATA_FOLDER=os.path.join("data",NAME_FOLDER)
N_STUDENTS = 25351 # Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30
MODEL_C = [0.01, 0.1, 1, 10]
N_TIME_WINDOWS = 5

def load_student_model(data_folder: str,mininteractions: int = 30,n_students: int = 100):
    
    print("!!!!!!!!!!!!!!!!Loading student model !!!!!!!!!!!!")
    pathMathiadata = os.path.join(data_folder, "..", NAME_FOLDER, "data.csv")
    stdmodel :SD.StudentDATA = SD.Mathiadata(pathMathiadata,seed=42)
    df ,Q= stdmodel.loadData(Display=True, min_intercation=mininteractions, n_students= n_students)
    df.to_csv(os.path.join(data_folder, f"preprocessed_data_{n_students}std.csv"), index=False)
    sparse.save_npz(os.path.join(data_folder, f"q_mat_{n_students}std.npz"), sparse.csr_matrix(Q))
    return stdmodel, df, Q

def prepare_features( data_folder,df,q_matrix: np.ndarray, stdmodel: SD.StudentDATA ):
    print("!!!!!!!!!!!!!!!!Preparing history !!!!!!!!!!!!")
    
    
    his = HIS.HistoryDATA(stdmodel=stdmodel)
    X, user_ids, item_ids, listKC = his.ComputeHistoryFeaturesTWKC(Q_mat=q_matrix, df=df)
    #save X to npz file
    sparse.save_npz(os.path.join(data_folder, f"history_features_{N_STUDENTS}std.npz"), sparse.csr_matrix(X))
    np.savez(os.path.join(data_folder, f"history_metadata_{N_STUDENTS}std.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listKC)
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
            os.path.join(data_folder, f"das3h_model_C{c}_{N_STUDENTS}std.pkl")
        )
    return modeldict

def plot_calibration1(y_true, y_pred_original):
    #plot la courbe de calibration du modèle original
    fig, ax = plt.subplots(figsize=(7, 7))
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_original, n_bins=10
    )   
    ax.plot(prob_pred, prob_true,
            marker='o', color="steelblue",
            linewidth=2, label="modèle original")
    ax.plot([0, 1], [0, 1],
            linestyle="--", color="gray",
            label="calibration parfaite")
    ax.fill_between(prob_pred,
                    prob_true - 0.05,
                    prob_true + 0.05,
                    alpha=0.1, color="green",
                    label="marge ±5%")
    ax.set_xlabel("P prédite")
    ax.set_ylabel("Taux réel de réussite")
    ax.legend()
    ax.grid(True, alpha=0.3)    


    plt.suptitle("Calibration — P prédite vs taux réel\n"
                 "Courbe proche diagonale = bien calibré \n"
                 "Dataset : MATHIA",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

def test_model_parameters(model: DAS3H.DAS3HModel):
    params = model.get_params()
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
    alpha_values = list(params["alpha_s"].values())
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

def plot_real_theta_vs_future(df, params_original):
    
    Window  = ["1h", "1j", "1sem", "1mois", "∞"]
    Timesec = [3600, 86400, 604800, 2592000, float("inf")]
    results = []

    for stud_id in df["user_id"].unique():
        df_stud = df[df["user_id"] == stud_id].sort_values("timestamp")

        for kc in df_stud["KC"].unique():
            df_kc = df_stud[df_stud["KC"] == kc]
            if len(df_kc) < 4:
                continue

            mid   = len(df_kc) // 2
            passe = df_kc.iloc[:mid]
            futur = df_kc.iloc[mid:]
            t_mid = passe["timestamp"].iloc[-1]

            # Features par fenêtre temporelle
            wins_tw     = np.zeros(len(Timesec))
            attempts_tw = np.zeros(len(Timesec))

            for tw, delta in enumerate(Timesec):
                if delta == float("inf"):
                    df_tw = passe
                else:
                    df_tw = passe[passe["timestamp"] >= t_mid - delta]
                wins_tw[tw]     = np.log(1 + df_tw["correct"].sum())
                attempts_tw[tw] = np.log(1 + len(df_tw))

            if kc in params_original["theta_wins"]:
                theta_wins_real     = np.dot(params_original["theta_wins"][kc],     wins_tw)
                theta_attempts_real = np.dot(params_original["theta_attempts"][kc], attempts_tw)
                sum_wins_attempts = theta_wins_real + theta_attempts_real
            else:
                theta_wins_real     = 0.0
                theta_attempts_real = 0.0
                sum_wins_attempts=0.0

            

            taux_futur = futur["correct"].mean()

            results.append({
                
                "History": sum_wins_attempts,
                "Future_Rate":          taux_futur
            })

    df_res = pd.DataFrame(results)

    # Corrélations
    print("=== Corrélations des VRAIS theta avec taux futur ===")
    for feat in ["History"]:
        print(f"{feat} : {df_res[feat].corr(df_res['Future_Rate']):.4f}")

    # Visualisation
    fig, axes = plt.subplots(1, 1, figsize=(18, 5))
    features = ["History"]
    titles = ["History vs Future rate",]

    for ax, feat, title in zip([axes], features, titles):
        df_res["bin"] = pd.cut(df_res[feat], bins=10)
        binned = df_res.groupby("bin")["Future_Rate"].agg(["mean", "std"])

        ax.plot(range(len(binned)), binned["mean"], marker='o')
        ax.fill_between(range(len(binned)),
                       binned["mean"] - binned["std"],
                       binned["mean"] + binned["std"],
                       alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel(f"Bins de {feat} ")
        ax.set_ylabel("Taux réussite futur moyen")
        ax.axhline(df_res["Future_Rate"].mean(), color="red",
                  linestyle="--", label="moyenne globale")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    return df_res
if __name__ == "__main__":
    time_execution = 4 #1: load and preprocess data, 2: prepare features, 0: both
    if time_execution==1:
        time_start = time.time()
        stdmodel,df,Q = load_student_model(DATA_FOLDER, MIN_INTERACTIONS, N_STUDENTS)
        time_end= time.time()
        print(f"Time to load and preprocess data: {time_end - time_start:.2f} seconds")
    elif time_execution==2:
        df= pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        Q = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        stdmodel = SD.Mathiadata()
        start_time = time.time()
        X, user_ids, item_ids, listKC = prepare_features(DATA_FOLDER, df, Q, stdmodel)
        end_time = time.time()
        print(f"Time to prepare features: {end_time - start_time:.2f} seconds") #35 min 
    elif time_execution==3:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        X=sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_{N_STUDENTS}std.npz"))
        metadata = np.load(os.path.join(DATA_FOLDER, f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)
        user_ids = metadata["user_ids"]
        item_ids = metadata["item_ids"]
        kc_list = metadata["kc_list"].tolist()
        start=time.time()
        modeldict = test_model_training(DATA_FOLDER,X, user_ids, item_ids, kc_list, model_c=MODEL_C, n_tw=N_TIME_WINDOWS, perc_init=0.2)
        end=time.time()
        print(f"Time to train models: {end - start:.2f} seconds") 

    elif time_execution==4:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))  
        X = sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_{N_STUDENTS}std.npz"))
        # Sélectionner le meilleur modèle selon AUC max, NLL et RMSE min
        best_auc = -float('inf')
        best_nll = float('inf')
        best_rmse = float('inf')
        best_score = -float('inf')
        best_model = None
        best_results = None
        best_c = None
        models_info = []
        for c in MODEL_C:
            model_path = os.path.join(DATA_FOLDER, f"das3h_model_C{c}_{N_STUDENTS}std.pkl")
            loaded = joblib.load(model_path)
            model = loaded["model"]
            results = loaded["results"]
            auc = results["AUC"]
            nll = results["NLL"]
            rmse = results["RMSE"]
            models_info.append((c, auc, nll, rmse, model, results))
            # Score composite : AUC élevé, NLL et RMSE faibles
            score = auc - nll - rmse
            if score > best_score:
                best_score = score
                best_auc = auc
                best_nll = nll
                best_rmse = rmse
                best_model = model
                best_results = results
                best_c = c
        print("Résultats pour chaque C :")
        for c, auc, nll, rmse, _, _ in models_info:
            print(f"C={c} : AUC={auc:.4f}, NLL={nll:.4f}, RMSE={rmse:.4f}")
        print(f"\nMeilleur modèle : C={best_c} | AUC={best_auc:.4f}, NLL={best_nll:.4f}, RMSE={best_rmse:.4f}")
        # Utiliser le meilleur modèle
        cols = list(range(X.shape[1]))
        cols.remove(3)
        X_no_col3 = X[:, cols]
        p_pred_raw = best_model.predict_proba(X_no_col3)
        p_pred = p_pred_raw if p_pred_raw.ndim == 1 else p_pred_raw[:, 1]
        y_true = X[:, 3].toarray().flatten()
        #plot_calibration1(y_true, p_pred)
        #params_original = test_model_parameters(best_model)
        df_res = plot_real_theta_vs_future(df, best_model.get_params())
        print("DOnnnne")
        
    print("!!!!Done!!!!!")