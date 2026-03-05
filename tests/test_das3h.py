

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

# =====================================================================
# CONFIGURATION
# =====================================================================
NAME_FOLDER="bridge_algebra06" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 1146 # Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30
MODEL_C = 0.01  # Regularization parameter
N_TIME_WINDOWS = 5


def setup_data(data_folder: str, n_students: int = 100):
    """Load preprocessed data and Q matrix"""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    csv_path = os.path.join(data_folder, f"preprocessed_data_{n_students}std.csv")
    q_matrix_path = os.path.join(data_folder, f"q_mat_{n_students}std.npz")
    
    df = pd.read_csv(csv_path, sep=",")
    q_matrix = sparse.load_npz(q_matrix_path).toarray()
    
    print(f"✓ Data shape: {df.shape}")
    print(f"✓ Q-matrix shape: {q_matrix.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    return df, q_matrix


def load_student_model(data_folder: str,mininteractions: int = 30,n_students: int = 100):
    
    print("!!!!!!!!!!!!!!!!Loading student model !!!!!!!!!!!!")
    pathbridge = os.path.join(data_folder, "..", NAME_FOLDER, "data.txt")
    stdmodel :SD.StudentDATA = SD.StudentDATA(file=pathbridge)
    df,Q=stdmodel.loadData(Display=False, min_intercation=mininteractions, n_students=n_students)
    #save df to csv and Q to npz
    df.to_csv(os.path.join(data_folder, f"preprocessed_data_{n_students}std.csv"), index=False)
    sparse.save_npz(os.path.join(data_folder, f"q_mat_{n_students}std.npz"), sparse.csr_matrix(Q))
    return stdmodel,df,Q


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



def plotScoreRByTime(df,NbInterTime=10):
     # c mieux pour gerer pd.cut 
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    delta = (end_time - start_time) / NbInterTime
    time_bins = [start_time + i * delta for i in range(NbInterTime + 1)]
    df["time_bin"] = pd.cut(df["timestamp"], bins=time_bins, labels=False, include_lowest=True)
    score_by_time = df.groupby("time_bin")["correct"].mean() #score moyen Nbreussite/NbreTentative
    plt.figure(figsize=(10, 5))
    #ajouter la variance et l'ecart type
    plt.fill_between(score_by_time.index, score_by_time.values - df.groupby("time_bin")["correct"].std(), 
                     score_by_time.values + df.groupby("time_bin")["correct"].std(), color="blue", alpha=0.2)
    plt.plot(score_by_time.index, score_by_time.values, color="blue")
    plt.xlabel("Time (days)")
    plt.ylabel("Score moyen (correct)")
    plt.title("Évolution du score moyen au cours du temps")
    bin_labels = [int(i * delta.days) for i in score_by_time.index]
    plt.xticks(score_by_time.index, bin_labels)
    plt.grid(True)
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



def plot_theta_vs_log_countsKC(df, params):
    Window  = ["1h", "1j", "1sem", "1mois", "∞"]
    Timesec = [3600, 86400, 604800, 2592000, float("inf")]
    kc_list = list(params["theta_wins"].keys())
    colors  = plt.cm.viridis(np.linspace(0, 1, len(Window)))
    t_max   = df["timestamp"].max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for tw, (label, delta) in enumerate(zip(Window, Timesec)):
        df_tw = df if delta == float("inf") else df[df["timestamp"] >= t_max - delta]

        wins_per_kc     = df_tw[df_tw["correct"] == 1].groupby("KC").size()
        attempts_per_kc = df_tw.groupby("KC").size()

        # Un point par KC
        x_w = np.array([np.log(1 + wins_per_kc.get(kc, 0))     for kc in kc_list])
        x_a = np.array([np.log(1 + attempts_per_kc.get(kc, 0)) for kc in kc_list])
        theta_w = np.array([params["theta_wins"][kc][tw]     for kc in kc_list])
        theta_a = np.array([params["theta_attempts"][kc][tw] for kc in kc_list])

        #axes[0].scatter(x_w, theta_w, alpha=0.4, s=15, color=colors[tw])
        #axes[1].scatter(x_a, theta_a, alpha=0.4, s=15, color=colors[tw])

        # Tendance moyenne par bin → vision globale
        for ax, x_vals, theta_vals in zip(axes, [x_w, x_a], [theta_w, theta_a]):
            sort_idx = np.argsort(x_vals)
            x_s, y_s = x_vals[sort_idx], theta_vals[sort_idx]
            bins = np.linspace(x_s.min(), x_s.max(), 10)

            b_means, b_stds, b_centers = [], [], []
            for b in range(len(bins) - 1):
                mask = (x_s >= bins[b]) & (x_s < bins[b+1])
                if mask.sum() == 0:
                    continue
                b_means.append(y_s[mask].mean())
                b_stds.append(y_s[mask].std())
                b_centers.append((bins[b] + bins[b+1]) / 2)

            b_means, b_stds, b_centers = map(np.array, [b_means, b_stds, b_centers])
            ax.plot(b_centers, b_means, color=colors[tw], linewidth=2, label=label)
            ax.fill_between(b_centers, b_means - b_stds, b_means + b_stds, alpha=0.1, color=colors[tw])

    axes[0].set(xlabel="log(1 + wins par KC)", ylabel="θ_wins", title="θ_wins vs log(1+wins) ")
    axes[1].set(xlabel="log(1 + attempts par KC)", ylabel="θ_attempts", title="θ_attempts vs log(1+attempts) ")
    for ax in axes:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(title="Fenêtre", fontsize=8)
        ax.grid(True)

    plt.suptitle("Theta vs log(1+count) par KC pour différentes fenêtres temporelles",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_theta_by_window(params):
    """Vérifier que theta_wins > 0 et theta_attempts > 0"""
    Window = ["1h", "1j", "1sem", "1mois", "∞"]
    
    mean_wins     = []
    mean_attempts = []
    std_wins      = []
    std_attempts  = []

    for k in range(5):
        w = [params["theta_wins"][kc][k]     for kc in params["theta_wins"]]
        a = [params["theta_attempts"][kc][k] for kc in params["theta_attempts"]]
        mean_wins.append(np.mean(w))
        mean_attempts.append(np.mean(a))
        std_wins.append(np.std(w))
        std_attempts.append(np.std(a))

    x = range(5)
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, mean_wins,     yerr=std_wins,color='blue',alpha=0.7,
                 marker='o', label='theta_wins',     capsize=5)
    plt.errorbar(x, mean_attempts, yerr=std_attempts,color='orange',alpha=0.5,
                 marker='s', label='theta_attempts', capsize=5)
    plt.xticks(x, Window)
    plt.xlabel("Fenêtre temporelle")
    plt.ylabel("Valeur moyenne du theta")
    plt.title("les Theta  pour chaque  fenêtre temporelle\n")
    plt.legend()
    plt.grid(True)
    plt.show()

def ComputeMeanTheta(params:dict=None):
    dicMeanThetaW= {}
    dicMeanThetaA= {}
    Window=[3600, 86400, 604800, 2592000, float("inf")]
    ThetaWins=params["theta_wins"]
    ThetaAttempts=params["theta_attempts"]
    for k in range(len(Window)):
        thetawinsK=[ThetaWins.get(kc)[k] for kc in ThetaWins.keys()]
        thetaattemptsK=[ThetaAttempts.get(kc)[k] for kc in ThetaAttempts.keys()]
        meanThetaW=np.mean(thetawinsK)
        meanThetaA=np.mean(thetaattemptsK)
        dicMeanThetaW[k]=meanThetaW
        dicMeanThetaA[k]=meanThetaA
    return dicMeanThetaW,dicMeanThetaA


def PlotMeanTheta(dicMeanThetaW:dict=None,dicMeanThetaA:dict=None):
    import matplotlib.pyplot as plt
    x = list(dicMeanThetaW.keys())
    y_wins = list(dicMeanThetaW.values())
    y_attempts = list(dicMeanThetaA.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_wins, marker='o', label='Mean Theta Wins')
    plt.plot(x, y_attempts, marker='o', label='Mean Theta Attempts')
    #plt.xscale('log')
    plt.xlabel('Time Window (seconds)')
    plt.ylabel('Mean Theta Value')
    plt.title('Mean Theta Wins and Attempts by Time Window')
    plt.xticks(x, ['1h', '1d', '1w', '30d', 'inf'])
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_wins_distribution(df, kc_list):
    """
    Analyse combien d'élèves ont des succès dans chaque fenêtre.
    """
    from collections import defaultdict
    TW_SECONDS = [3600, 86400, 604800, 2592000, float("inf")]
    TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
    
    counts_by_tw = {tw: [] for tw in TW_LABELS}
    
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].sort_values("timestamp")
        t_current = df_user["timestamp"].max()
        
        wins_timestamps = defaultdict(list)
        
        for _, row in df_user.iterrows():
            if row["correct"] == 1:
                kcs_row = str(row["KC"]).split("~~")
                for kc in kcs_row:
                    if kc in kc_list:
                        wins_timestamps[kc].append(row["timestamp"])
        
        # Compter par fenêtre
        for kc in kc_list:
            if kc not in wins_timestamps:
                continue
            
            timestamps = wins_timestamps[kc]
            for i, (tw_sec, tw_label) in enumerate(zip(TW_SECONDS, TW_LABELS)):
                if tw_sec == float("inf"):
                    n = len(timestamps)
                else:
                    n = sum(1 for t in timestamps if (t_current - t) <= tw_sec)
                
                counts_by_tw[tw_label].append(n)
    
    # Afficher
    print(f"\n{'='*70}")
    print(f"  Distribution des succès par fenêtre temporelle")
    print(f"{'='*70}\n")
    print(f"{'Fenêtre':<10} {'Total':<12} {'Moyenne':<12} {'Médiane':<12} {'% avec n>0':<15}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*15}")
    
    for tw_label in TW_LABELS:
        counts = counts_by_tw[tw_label]
        sum_=np.sum(counts)
        avg = np.mean(counts)
        med = np.median(counts)
        pct_nonzero = sum(1 for c in counts if c > 0) / len(counts) * 100
        
        print(f"{tw_label:<10} {sum_:<12.0f} {avg:>10.2f}   {med:>10.0f}   {pct_nonzero:>13.1f}%")
    
    print(f"{'='*70}\n")

def analyse_eleve(df, X, user, params, model):
    print("The user is :", user)
    print("The level of ability of the user is :", params["alpha_s"].get(user))

    #Difficulty 
    df_u = df[df.user_id == user][["item_id", "correct", "KC", "timestamp"]]
    delta = params["delta_j"] 
    df_compare = df_u.copy()
    df_compare["delta_j"] = df_compare["item_id"].map(delta)
    df_compare.sort_values("delta_j")
    df_compare["bien_estime"] = (
    ((df_compare["delta_j"] > 0) & (df_compare["correct"] == 0)) |
    ((df_compare["delta_j"] < 0) & (df_compare["correct"] == 1)))
    nb_bien = df_compare["bien_estime"].sum()
    nb_total = len(df_compare)
    nb_mal = nb_total - nb_bien
    print(f"\n{'='*70}")
    print("ANALYSE DE L'item")
    print(f"{'='*70}\n")
    print("Bien estimés :", nb_bien)
    print("Mal estimés  :", nb_mal)
    print("Taux de bonne estimation :", nb_bien / nb_total)
    plt.figure(figsize=(10,5))

    plt.scatter(df_compare["delta_j"],df_compare["correct"],c=df_compare["bien_estime"].map({True: "green", False: "red"}),alpha=0.7,s=80)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.6)
    plt.xlabel("delta_j (difficulté estimée)")
    plt.ylabel("correct (0 = échec, 1 = réussite)")
    plt.title("Items bien estimés (vert) vs mal estimés (rouge)")
    plt.show()

    #Kc list 

    df_u["KC_list"] = df_u["KC"].astype(str).str.split("~~")
    df_u = df_u.drop(columns=["KC"])
    df_exploded = df_u.explode("KC_list").rename(columns={"KC_list": "KC"})
    kc_perf = df_exploded.groupby("KC")["correct"].mean()
    params = model.get_params()
    beta = params["beta_k"]

    df_kc = kc_perf.to_frame("accuracy")
    df_kc["beta_k"] = df_kc.index.map(beta)
    df_kc["bien_estime"] = (
        ((df_kc["beta_k"] > 0) & (df_kc["accuracy"] >= 0.5)) |
        ((df_kc["beta_k"] < 0) & (df_kc["accuracy"] < 0.5))
    )
    nb_bien = df_kc["bien_estime"].sum()
    nb_total = len(df_kc)
    nb_mal = nb_total - nb_bien
    print(f"\n{'='*70}")
    print("ANALYSE DES COMPETENCES (KC)")
    print(f"{'='*70}\n")
    print("Bien estimés :", nb_bien)
    print("Mal estimés  :", nb_mal)
    print("Taux de bonne estimation :", nb_bien / nb_total)

    plt.figure(figsize=(8,5))
    plt.scatter(
        df_kc["beta_k"],
        df_kc["accuracy"],
        c=df_kc["bien_estime"].map({True: "green", False: "red"}),
        s=80
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("beta_k (facilité estimée du KC)")
    plt.ylabel("Accuracy brute de l'élève")
    plt.title("KC bien estimés (vert) vs mal estimés (rouge)")
    plt.show()


def PlotPrVar(df,NbInterTime=10,chose=1):
    if chose==1:
        df = df.sort_values(["user_id", "KC", "timestamp"])
        # Utilisation de groupby + shift pour vectoriser le calcul de delta_t
        df["delta_t"] = df.groupby(["user_id", "KC"])['timestamp'].diff()
        df = df.dropna(subset=["delta_t"])
        time_bins = [
            0,
            3600,
            3600*24,
            3600*24*7,
            3600*24*30,
            df["delta_t"].max()
        ]
        bin_labels = ["<1h", "1h-1j", "1j-7j", "7j-30j", ">30j"]
    else : 
        df["delta_t"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("delta_t")
        start_time = df["delta_t"].min()
        end_time = df["delta_t"].max()
        delta = (end_time - start_time) / NbInterTime
        time_bins = [start_time + i * delta for i in range(NbInterTime + 1)]
        bin_labels = [int(i * delta.days) for i in range(NbInterTime)]
        
    df["time_bin"] = pd.cut(df["delta_t"], bins=time_bins, labels=bin_labels, include_lowest=True)
    curve = df.groupby("time_bin")["correct"].mean()
    plt.plot(curve.index.astype(str), curve.values,color="red", linewidth=2)
    plt.fill_between(curve.index.astype(str), curve.values - df.groupby("time_bin")["correct"].std(),
                     curve.values + df.groupby("time_bin")["correct"].std(), color="red", alpha=0.2)
    plt.xticks(rotation=45)
    plt.ylabel("Sucess rate (correct)")
    if chose==1:
        plt.xlabel("Time since last revision")
    else:
        plt.xlabel("Time (days)")
    plt.title("Forgetting curve for all students")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
#On prends les 3 meilleurs KC et on regarde pour tous les etudiants 
def PlotPrVarByKC(df, NbInterTime=10,chose=1):
    if chose==1:
        df = df.sort_values(["user_id", "KC", "timestamp"])
        # Utilisation de groupby + shift pour vectoriser le calcul de delta_t
        df["delta_t"] = df.groupby(["user_id", "KC"])['timestamp'].diff()
        df = df.dropna(subset=["delta_t"])
        time_bins = [
            0,
            3600,
            3600*24,
            3600*24*7,
            3600*24*30,
            df["delta_t"].max()
        ]
        bin_labels = ["<1h", "1h-1j", "1j-7j", "7j-30j", ">30j"]
    else :
        df["delta_t"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("delta_t")
        start_time = df["delta_t"].min()
        end_time = df["delta_t"].max()
        delta = (end_time - start_time) / NbInterTime
        time_bins = [start_time + i * delta for i in range(NbInterTime + 1)]
        bin_labels = [int(i * delta.days) for i in range(NbInterTime)]
    
    df["time_bin"] = pd.cut(df["delta_t"], bins=time_bins, labels=bin_labels, include_lowest=True)
    
    top_kcs = df["KC"].value_counts().nlargest(5).index
    plt.figure(figsize=(12, 6))
    
    # Palette automatique pour n'importe quel nombre de KC
    cmap = plt.cm.get_cmap('rainbow', len(top_kcs))
    for i, kc in enumerate(top_kcs):
        color = cmap(i)
        df_kc = df[df["KC"] == kc]
        curve = df_kc.groupby("time_bin")["correct"].mean()
        plt.plot(curve.index.astype(str), curve.values, marker='o', label=f"KC : {kc}", color=color)
        plt.fill_between(curve.index.astype(str), curve.values - df_kc.groupby("time_bin")["correct"].std(),
                         curve.values + df_kc.groupby("time_bin")["correct"].std(), alpha=0.15, color=color)
    
    plt.xticks(rotation=45)
    plt.ylabel("Sucess rate (correct)")
    if chose==1:
        plt.xlabel("Time since last revision")
    else:
        plt.xlabel("Time (days)")
    plt.title(f"Forgetting curve by KC for top {len(top_kcs)} KCs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

#Compute proba Das3h for all user and item and plot the mean proba and correct for bin of time 
def ComputeProbaOverTime1(df, model, UserOrItem, choseItemOrUser=1):

    AllProbas   = []
    AllCorrects = []
    timestamps  = []

    if choseItemOrUser == 1:
        df_ItemUser = df[df["item_id"] == UserOrItem].sort_values("timestamp")
    elif choseItemOrUser == 2:
        df_ItemUser = df[df["user_id"] == UserOrItem].sort_values("timestamp")
    else:
        df_ItemUser = df.sort_values("timestamp")

    q = defaultdict(OurQueue)
    
    fails_per_user_kc = defaultdict(int)

    for _, row in df_ItemUser.iterrows():
        if choseItemOrUser == 1:
            user_id = row["user_id"]
            item_id = UserOrItem
        elif choseItemOrUser == 2:
            item_id = row["item_id"]
            user_id = UserOrItem
        else:
            user_id = row["user_id"]
            item_id = row["item_id"]
        

        kc_list = str(row["KC"]).split("~~")
        t       = row["timestamp"]
        correct = int(row["correct"])

        history = {}
        for kc in kc_list:
            history[kc] = {
                "wins":     np.log(1 + np.array(q[user_id, kc, "correct"].get_counters(t))),
                "attempts": np.log(1 + np.array(q[user_id, kc].get_counters(t))),
                "fails":    fails_per_user_kc[user_id, kc]
            }

        p = model.predict_single(
            user_id=user_id,
            item_id=item_id,
            kc_list=kc_list,
            history=history
        )

        AllProbas.append(p)
        AllCorrects.append(correct)
        timestamps.append(t)

        for kc in kc_list:
            q[user_id, kc].push(t)
            if correct:
                q[user_id, kc, "correct"].push(t)
            else:
                fails_per_user_kc[user_id, kc] += 1

    return pd.DataFrame({
        "timestamp": timestamps,
        "proba":     AllProbas,
        "correct":   AllCorrects
    })

def ComputeProbaOverTime(df, model, UserOrItem=None, choseItemOrUser=1, 
                          max_per_user=10):
    """
    choseItemOrUser : 1=par item, 2=par élève, 3=global
    max_per_user    : pour cas 3, max interactions par élève
    sample_frac     : pour cas 3, fraction du dataset (ex: 0.1 = 10%)
    """
    AllProbas   = []
    AllCorrects = []
    timestamps  = []

    # --- Sélection et échantillonnage ---
    if choseItemOrUser == 1:
        df_ItemUser = df[df["item_id"] == UserOrItem].sort_values("timestamp")

    elif choseItemOrUser == 2:
        df_ItemUser = df[df["user_id"] == UserOrItem].sort_values("timestamp")

    else:
        df_ItemUser = df.sort_values("timestamp")\
                        .groupby("user_id")\
                        .head(max_per_user)\
                        .sort_values("timestamp")
        print(f"Échantillon : {len(df_ItemUser)} interactions "
                f"(max {max_per_user} par élève)")

    # --- Convertir en numpy pour vitesse ---
    data = df_ItemUser[["user_id", "item_id", "timestamp", 
                         "correct", "KC"]].to_numpy()
    total = len(data)

    q                 = defaultdict(OurQueue)
    fails_per_user_kc = defaultdict(int)

    for l in range(total):

        
        print(f"  {l}/{total} ({l/total*100:.0f}%)")

        user_id = int(data[l, 0])
        item_id = int(data[l, 1])
        t       = float(data[l, 2])
        correct = int(data[l, 3])
        kc_list = str(data[l, 4]).split("~~")

        # Overrides pour cas 1 et 2
        if choseItemOrUser == 1:
            item_id = UserOrItem
        elif choseItemOrUser == 2:
            user_id = UserOrItem

        history = {}
        for kc in kc_list:
            history[kc] = {
                "wins":     np.log(1 + np.array(
                                q[user_id, kc, "correct"].get_counters(t))),
                "attempts": np.log(1 + np.array(
                                q[user_id, kc].get_counters(t))),
                "fails":    fails_per_user_kc[user_id, kc]
            }

        p = model.predict_single(
            user_id=user_id,
            item_id=item_id,
            kc_list=kc_list,
            history=history
        )

        AllProbas.append(p)
        AllCorrects.append(correct)
        timestamps.append(t)

        for kc in kc_list:
            q[user_id, kc].push(t)
            if correct:
                q[user_id, kc, "correct"].push(t)
            else:
                fails_per_user_kc[user_id, kc] += 1

    return pd.DataFrame({
        "timestamp": timestamps,
        "proba":     AllProbas,
        "correct":   AllCorrects
    })
def PlotMAsterOverTimeByItem(df, UserORItem,NbInterTime=10,chose=1):
    #On va regrouper par bin de temps et faire la moyenne de la proba et de correct pour chaque bin
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    delta = (end_time - start_time) / NbInterTime
    time_bins = [start_time + i * delta for i in range(NbInterTime + 1)]
    bins_labels = [int(i * delta.days) for i in range(NbInterTime)]
    df["time_bin"] = pd.cut(df["timestamp"], bins=time_bins, labels=False, include_lowest=True)
    meanPdash = df.groupby("time_bin")["proba"].mean()
    meanPN= df.groupby("time_bin")["correct"].mean()
    plt.figure(figsize=(10, 5))
    
    plt.plot(meanPdash.index, meanPdash.values, marker='o', label='Predict Proba(DAS3H)', color="blue")
    plt.fill_between(meanPdash.index, meanPdash.values - df.groupby("time_bin")["proba"].std(), 
                     meanPdash.values + df.groupby("time_bin")["proba"].std(), color="blue", alpha=0.2)
    plt.plot(meanPN.index, meanPN.values, marker='s', label='Real Success Rate', color="orange")
    plt.fill_between(meanPN.index, meanPN.values - df.groupby("time_bin")["correct"].std(), 
                     meanPN.values + df.groupby("time_bin")["correct"].std(), color="orange", alpha=0.2)
    plt.xticks(meanPdash.index, bins_labels[:len(meanPdash.index)])
    plt.xlabel("Time (days)")
    plt.ylabel("Proba / Real Success Rate")
    if chose==1:
        plt.title(f"Mastery over time for item {UserORItem}")
    elif chose==2:
        plt.title(f"Success rate over time for student {UserORItem}")
    else : 
        plt.title(f"Mastery over time for all data")
        
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    timetoexeucte=3 # 0 pour df et Q , 1 pour les features, 2 pour le modèle , 3 pour les param et la visua 

    #stdmodel,df,Q=load_student_model(DATA_FOLDER, MIN_INTERACTIONS, N_STUDENTS)
    if timetoexeucte==0:
        start=time.time()
        stdmodel,df,Q=load_student_model(DATA_FOLDER, MIN_INTERACTIONS, N_STUDENTS)
        end=time.time()
        print(f"Time to load and preprocess data: {end - start:.2f} seconds")
    elif timetoexeucte==1:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        start=time.time()
        X, user_ids, item_ids, kc_list = prepare_features(DATA_FOLDER,df,q_matrix, stdmodel=None)
        end=time.time()
        print(f"Time to prepare features: {end - start:.2f} seconds")
    elif timetoexeucte==2:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        X=sparse.load_npz(os.path.join(DATA_FOLDER, f"history_features_{N_STUDENTS}std.npz"))
        metadata = np.load(os.path.join(DATA_FOLDER, f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)
        user_ids = metadata["user_ids"]
        item_ids = metadata["item_ids"]
        kc_list = metadata["kc_list"].tolist()
        start=time.time()
        modeldict = test_model_training(DATA_FOLDER,X, user_ids, item_ids, kc_list, model_c=[0.01, 0.1, 1.0], n_tw=N_TIME_WINDOWS, perc_init=0.2)
        end=time.time()
        print(f"Time to train models: {end - start:.2f} seconds")
    elif timetoexeucte==3:
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        #plotScoreRByTime(df,NbInterTime=50)
        #PlotPrVar(df,NbInterTime=50,chose=1)
        #PlotPrVar(df,NbInterTime=50,chose=2)
        #PlotPrVarByKC(df, NbInterTime=50,chose=1)
        #PlotPrVarByKC(df, NbInterTime=50,chose=2)
        model_path = os.path.join(DATA_FOLDER, f"das3h_model_C{1.0}_{N_STUDENTS}std.pkl")
        loaded = joblib.load(model_path)
        model = loaded["model"]
        #chose the most practiced item
        #On va choisir des items et user aleatoires (5 et on regarde)
        start=time.time()
        df_probaAll=ComputeProbaOverTime(df, model, UserOrItem=None,choseItemOrUser=3,max_per_user=1000)
        end=time.time()
        print(f"Time to compute probabilities over time: {(end - start)/60:.2f} minutes")
        PlotMAsterOverTimeByItem(df_probaAll, UserORItem=None,NbInterTime=10,chose=3)
        #user_id = df["user_id"].value_counts().idxmax()
        #item_id = df["item_id"].value_counts().idxmax()
        #df_proba = ComputeProbaOverTime(df, model, UserOrItem=item_id,choseItemOrUser=1)
        #PlotMAsterOverTimeByItem(df_proba, UserORItem=item_id,NbInterTime=10,chose=1)
        results = loaded["results"]
        params = test_model_parameters(model)
        #plot_theta_vs_log_counts(df, params=params)
        plot_theta_vs_log_countsKC(df, params=params)
    
   
    print("!!!!!!!!!!!!!! Done ^^  !!!!!!!!!!!!!")
   