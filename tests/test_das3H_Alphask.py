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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
NAME_FOLDER="bridge_algebra06" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 1146 # Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30
MODEL_C = 0.01  # Regularization parameter
N_TIME_WINDOWS = 5

def prepare_featuresRatioAlpha( data_folder,df,q_matrix: np.ndarray, stdmodel: SD.StudentDATA ):
    print("!!!!!!!!!!!!!!!!Preparing history !!!!!!!!!!!!")
    
    
    his = HIS.HistoryDATA(stdmodel=stdmodel)
    X, user_ids, item_ids, listKC = his.ComputeHistoryfeaturesRatioAlpha(Q_mat=q_matrix, df=df)
    #save X to npz file
    sparse.save_npz(os.path.join(data_folder, f"history_features_Ratio_Alpha{N_STUDENTS}std.npz"), sparse.csr_matrix(X))
    np.savez(os.path.join(data_folder, f"history_metadata_Ratio_Alpha{N_STUDENTS}std.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listKC)
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
            os.path.join(data_folder, f"das3h_model_RatioAlpha_C{c}_{N_STUDENTS}std.pkl")
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

def plot_calibration(y_true, y_pred_original, y_pred_alphask):
    """
    Si P=0.7 → l'élève doit réussir 70% du temps en réalité
    C'est ça la calibration
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_original, y_pred_alphask],
        ["Modèle original (alpha_s)", "Modèle theta_ratio "]
    ):
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred, n_bins=10
        )

        ax.plot(prob_pred, prob_true,
                marker='o', color="steelblue",
                linewidth=2, label="modèle")
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
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Calibration — P prédite vs taux réel\n"
                 "Courbe proche diagonale = bien calibré ",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_real_theta_vs_future(df, params_original, params_ratio):
    
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

            ratio_tw = wins_tw - attempts_tw

            
            if kc in params_original["theta_wins"]:
                theta_wins_real     = np.dot(params_original["theta_wins"][kc],     wins_tw)
                theta_attempts_real = np.dot(params_original["theta_attempts"][kc], attempts_tw)
                sum_wins_attempts = theta_wins_real + theta_attempts_real
            else:
                theta_wins_real     = 0.0
                theta_attempts_real = 0.0
                sum_wins_attempts=0.0

            if kc in params_ratio["theta_ratio"]:
                theta_ratio_real = np.dot(params_ratio["theta_ratio"][kc], ratio_tw)
            else:
                theta_ratio_real = 0.0

            taux_futur = futur["correct"].mean()

            results.append({
                "theta_wins_log":     theta_wins_real,
                "theta_attempts_log": theta_attempts_real,
                "theta_sum": sum_wins_attempts,
                "theta_ratio_log":    theta_ratio_real,
                "taux_futur":          taux_futur
            })

    df_res = pd.DataFrame(results)

    # Corrélations
    print("=== Corrélations des VRAIS theta avec taux futur ===")
    for feat in ["theta_wins_log", "theta_attempts_log","theta_sum", "theta_ratio_log"]:
        print(f"{feat} : {df_res[feat].corr(df_res['taux_futur']):.4f}")

    # Visualisation
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    features = ["theta_wins_log", "theta_attempts_log","theta_sum", "theta_ratio_log"]
    titles = ["θ_wins log(1+wins) vs taux futur",
          "θ_attempts log(1+attempts) vs taux futur",
          "θ_wins log(1+wins) + θ_attempts log(1+attempts) vs futur",  
          "θ_ratio log((1+w)/(1+a)) vs taux futur"]

    for ax, feat, title in zip(axes, features, titles):
        df_res["bin"] = pd.cut(df_res[feat], bins=10)
        binned = df_res.groupby("bin")["taux_futur"].agg(["mean", "std"])

        ax.plot(range(len(binned)), binned["mean"], marker='o')
        ax.fill_between(range(len(binned)),
                       binned["mean"] - binned["std"],
                       binned["mean"] + binned["std"],
                       alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel(f"Bins de {feat} ")
        ax.set_ylabel("Taux réussite futur moyen")
        ax.axhline(df_res["taux_futur"].mean(), color="red",
                  linestyle="--", label="moyenne globale")
        ax.legend()
        ax.grid(True)

    plt.suptitle("Relation entre les VRAIS theta appris et la réussite future",
                fontweight="bold")
    plt.tight_layout()
    plt.show()

    return df_res
def plot_log_vs_future_sucess(df, params_original, params_ratio):
    """
    Pour chaque élève x KC, on coupe l'historique en deux :
    - passé  → calcule theta
    - futur  → calcule taux réel de réussite
    Puis on trace theta vs taux futur
    """
    results = []

    for stud_id in df["user_id"].unique():
        df_stud = df[df["user_id"] == stud_id].sort_values("timestamp")
        
        for kc in df_stud["KC"].unique():
            df_kc = df_stud[df_stud["KC"] == kc]
            if len(df_kc) < 4:  # besoin d'assez d'historique
                continue
            
            mid = len(df_kc) // 2
            passe = df_kc.iloc[:mid]
            futur = df_kc.iloc[mid:]
            
            w = passe["correct"].sum()
            a = len(passe)
            
            log_wins    = np.log(1 + w)
            log_attempts = np.log(1 + a)
            log_ratio   = np.log((1 + w) / (1 + a))
            taux_futur        = futur["correct"].mean()
            
            results.append({
                "log_wins":     log_wins,
                "log_attempts": log_attempts,
                "log_ratio":    log_ratio,
                "taux_futur":     taux_futur
            })

    df_res = pd.DataFrame(results)

    # Corrélations
    print("=== Corrélations avec taux de réussite futur ===")
    print(f"log(1+wins)     : {df_res['log_wins'].corr(df_res['taux_futur']):.4f}")
    print(f"log(1+attempts) : {df_res['log_attempts'].corr(df_res['taux_futur']):.4f}")
    print(f"log((1+wins)/(1+attempts))   : {df_res['log_ratio'].corr(df_res['taux_futur']):.4f}")

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    features = ["log_wins", "log_attempts", "log_ratio"]
    titles   = ["log(1+wins) vs taux futur", "log(1+attempts) vs taux futur", "log((1+wins)/(1+attempts)) vs taux futur"]

    for ax, feat, title in zip(axes, features, titles):
        # Binning pour voir la tendance
        df_res["bin"] = pd.cut(df_res[feat], bins=10)
        binned = df_res.groupby("bin")["taux_futur"].agg(["mean", "std"])
        
        ax.plot(range(len(binned)), binned["mean"], marker='o')
        ax.fill_between(range(len(binned)),
                       binned["mean"] - binned["std"],
                       binned["mean"] + binned["std"],
                       alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel(f"Bins de {feat} (croissant )")
        ax.set_ylabel("Taux réussite futur moyen")
        ax.axhline(df_res["taux_futur"].mean(), color="red",
                  linestyle="--", label="moyenne globale")
        ax.legend()
        ax.grid(True)

    plt.suptitle("Relation l'historique et le futur",
                fontweight="bold")
    plt.tight_layout()
    plt.show()


def compare_discrimination(df, params_original, params_alphask):

    users     = df["user_id"].unique()
    taux_reel = df.groupby("user_id")["correct"].mean()

    alpha_original = [
        params_original["alpha_s"].get(u, 0.0)
        for u in users
    ]

    alpha_by_user = defaultdict(list)
    for (user_id, kc), val in params_alphask["alpha_sk"].items():
        alpha_by_user[user_id].append(val)

    alpha_sk_mean = [
        np.mean(alpha_by_user[u]) if alpha_by_user[u] else 0.0
        for u in users
    ]

    taux = [taux_reel.get(u, 0.5) for u in users]

    corr_original = np.corrcoef(alpha_original, taux)[0, 1]
    corr_alphask  = np.corrcoef(alpha_sk_mean,  taux)[0, 1]

    print(f"Corrélation alpha_s  vs taux réel : {corr_original:.3f}")
    print(f"Corrélation alpha_sk vs taux réel : {corr_alphask:.3f}")

    if corr_alphask > corr_original:
        print("alpha_sk discrimine mieux les élèves")
    else:
        print("alpha_s original discrimine mieux")

    # Graphe
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, alpha, title, corr in zip(
        axes,
        [alpha_original, alpha_sk_mean],
        ["alpha_s original", "alpha_sk moyen"],
        [corr_original, corr_alphask]
    ):
        ax.scatter(alpha, taux, alpha=0.4, s=20, color="steelblue")
        z      = np.polyfit(alpha, taux, 1)
        x_line = np.linspace(min(alpha), max(alpha), 100)
        ax.plot(x_line, np.poly1d(z)(x_line),
                color="red", linewidth=2)
        ax.set_xlabel("Alpha estimé")
        ax.set_ylabel("Taux réel de réussite")
        ax.set_title(f"{title}\nr = {corr:.3f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_discrimination_by_kc(df, params_alphask, top_n_kc=10):
    """
    Pour chaque KC :
    - calculer alpha_sk de chaque élève
    - comparer avec son taux réel sur ce KC
    - si corrélation forte → alpha_sk est informatif
    """
    kc_counts = df["KC"].value_counts().nlargest(top_n_kc).index
    correlations = {}

    for kc in kc_counts:
        df_kc = df[df["KC"].str.contains(kc, regex=False)]
        taux_par_user = df_kc.groupby("user_id")["correct"].mean()

        alpha_vals = []
        taux_vals  = []
        for u, taux in taux_par_user.items():
            a = params_alphask["alpha_sk"].get((u, kc), None)
            if a is not None:
                alpha_vals.append(a)
                taux_vals.append(taux)

        if len(alpha_vals) > 10:
            corr = np.corrcoef(alpha_vals, taux_vals)[0, 1]
            correlations[kc] = corr

    # Tracer
    kcs   = list(correlations.keys())
    corrs = list(correlations.values())

    plt.figure(figsize=(12, 6))
    colors = ["green" if c > 0.3 else
              "orange" if c > 0.1 else "red"
              for c in corrs]
    plt.barh([k[:40] for k in kcs], corrs, color=colors)
    plt.axvline(0.3, color="green",  linestyle="--",
                label="seuil bon (0.3)")
    plt.axvline(0.1, color="orange", linestyle="--",
                label="seuil acceptable (0.1)")
    plt.xlabel("Corrélation alpha_sk / taux réel par KC")
    plt.title("Est-ce que alpha_sk est informatif par KC ?\n"
              "Vert = bon, Orange = faible, Rouge = inutile")
    plt.legend()
    plt.tight_layout()
    plt.savefig("discrimination_by_kc.png", dpi=150)
    plt.show()

    print("\nCorrélation moyenne : "
          f"{np.mean(list(correlations.values())):.3f}")
    print("KCs bien discriminés (r>0.3) : "
          f"{sum(1 for c in corrs if c > 0.3)}/{len(corrs)}")
if __name__ == "__main__":
    timetoexeucte=3 #Time to execute

    if timetoexeucte==1: 
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        start=time.time()
        X, user_ids, item_ids, kc_list = prepare_featuresRatioAlpha(DATA_FOLDER,df,q_matrix, stdmodel=None)
        end=time.time()
        print(f"Time to prepare features: {end - start:.2f} seconds")
    elif timetoexeucte==2:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        X=sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_Ratio{N_STUDENTS}std.npz"))
        metadata = np.load(os.path.join(DATA_FOLDER, f"history_metadata_Ratio{N_STUDENTS}std.npz"), allow_pickle=True)
        user_ids = metadata["user_ids"]
        item_ids = metadata["item_ids"]
        kc_list = metadata["kc_list"]
        
        start=time.time()
        modeldict = test_model_training(DATA_FOLDER,X, user_ids, item_ids, kc_list, model_c=[0.01,0.1,1], n_tw=N_TIME_WINDOWS, perc_init=0.2)
        end=time.time()
        print(f"Time to train model: {end - start:.2f} seconds")
    elif timetoexeucte==3:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))        
        model_path_Aalphask = os.path.join(DATA_FOLDER, f"das3h_model_Ratio_C0.1_{N_STUDENTS}std.pkl")
        model_path_original = os.path.join(DATA_FOLDER, f"das3h_model_C1.0_{N_STUDENTS}std.pkl")
        X_alphask = sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_Ratio{N_STUDENTS}std.npz"))
        X_original = sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_{N_STUDENTS}std.npz"))
        loaded = joblib.load(model_path_Aalphask)
        model = loaded["model"]
        loaded_original = joblib.load(model_path_original)
        model_original = loaded_original["model"]
        #params_original = test_model_parameters(model_original)
        cols_original = list(range(X_original.shape[1]))
        cols_original.remove(3)
        X_original_no_col3 = X_original[:, cols_original]
        p_original=model_original.model.predict_proba(X_original_no_col3)[:, 1]
        #remove colonne 3 de X_alphask pour faire la prediction
        cols_alphask = list(range(X_alphask.shape[1]))
        cols_alphask.remove(3)
        X_alphask_no_col3 = X_alphask[:, cols_alphask]
        p_alphask=model.model.predict_proba(X_alphask_no_col3)[:, 1]
        y_true = X_alphask[:, 3].toarray().flatten()    
        plot_real_theta_vs_future(df,model_original.get_params(),model.get_paramsRatio())
        #plot_log_vs_future_sucess(df,model_original.get_params(),model.get_paramsRatio())
        #plot_calibration(y_true, p_original, p_alphask)
        

    print("!!!!!!!!!!!!Done!!!!!!!!!!!!!!!!!!!")