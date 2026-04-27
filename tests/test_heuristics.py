import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import src.Process.Heuristics.mu_back as MuBackH
import src.Process.Heuristics.Random as RandomH
import pandas as pd
from scipy import sparse
import numpy as np 
import src.Process.Simulation.SimuH as SimuH
import src.Process.Heuristics.no_review as Noreview
import src.Process.Heuristics.Theta_TresholdH as Theta_TresholdH
import src.datamodel.GraphSkills as Graph 

NAME_FOLDER="Mathiadata3" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 25351
import matplotlib.pyplot as plt
def load_Model():
    # Load the model
    df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    kc_list=np.load(os.path.join(DATA_FOLDER,f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)["kc_list"]
    
    return df,q_matrix,kc_list

def plot_pmr(all_results_pmr):
    plt.figure(figsize=(10, 6))
    for name, pmr in all_results_pmr.items():
        means = [pmr[w]["mean"] for w in sorted(pmr.keys())]
        stds = [pmr[w]["std"] for w in sorted(pmr.keys())]
        plt.plot(means, label=name)
        plt.fill_between(range(len(means)), np.array(means)-np.array(stds),
                         np.array(means)+np.array(stds), alpha=0.2)
    plt.xlabel("Semaine")
    plt.ylabel("PMR moyen")
    plt.title("Comparaison des heuristiques — PMR moyen par semaine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
def plot_mastery(all_results_mastery):
    plt.figure(figsize=(10, 6))
    for name, mastery in all_results_mastery.items():
        means = [mastery[w]["mean"] for w in sorted(mastery.keys())]
        stds = [mastery[w]["std"] for w in sorted(mastery.keys())]
        plt.plot(means, label=name)
        plt.fill_between(range(len(means)), np.array(means)-np.array(stds),
                         np.array(means)+np.array(stds), alpha=0.2)
    plt.xlabel("Semaine")
    plt.ylabel("% KCs maîtrisés (PMR ≥ 0.7)")
    plt.title("Comparaison des heuristiques — Taux de maîtrise par semaine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def aggregate_runs(all_runs):
    aggregated = {}
    for name, runs in all_runs.items():
        weeks=sorted(runs[0].keys())
        means_per_run=np.array([[run[w]["mean"] for w in weeks] for run in runs])
        stds_per_run=np.array([[run[w]["std"] for w in weeks] for run in runs])
        aggregated[name] = {
           "weeks": weeks,
              "mean": np.mean(means_per_run, axis=0),
                "std": np.mean(stds_per_run, axis=0)
        }
    return aggregated

def plot_aggregated(agg,ylabel,title,n_runs=10,xlabel="Semaine"):
    plt.figure(figsize=(10, 6))
    for name, data in agg.items():
        weeks = data["weeks"]
        means = data["mean"]
        stds = data["std"]
        plt.plot(weeks, means, label=name)
        #plt.fill_between(weeks, means - stds, means + stds, alpha=0.2)
    plt.axvline(16, color='gray', linestyle='--', label="Fin de l'apprentissage")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} ({n_runs} runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
def plot_pmr_par_groupe(df_pmr, title=""):
    # Séparer en 3 groupes selon alpha_s
    median_alpha = df_pmr["alpha_s"].median()
    q25 = df_pmr["alpha_s"].quantile(0.25)
    q75 = df_pmr["alpha_s"].quantile(0.75)
    
    df_pmr["groupe"] = "moyen"
    df_pmr.loc[df_pmr["alpha_s"] <= q25, "groupe"] = "faible"
    df_pmr.loc[df_pmr["alpha_s"] >= q75, "groupe"] = "fort"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for groupe, color in [("faible", "red"), ("moyen", "orange"), ("fort", "green")]:
        df_g = df_pmr[df_pmr["groupe"] == groupe]
        means = df_g.groupby("week")["pmr"].mean()
        ax.plot(means.index, means.values, label=f"Élèves {groupe}", color=color)
    
    ax.set_xlabel("Semaine")
    ax.set_ylabel("PMR moyen")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_variance_pmr(df_pmr_das3h, df_pmr_sans):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    var_das3h = df_pmr_das3h.groupby("week")["pmr"].std()
    var_sans = df_pmr_sans.groupby("week")["pmr"].std()
    
    ax.plot(var_das3h.index, var_das3h.values, label="Avec DAS3H", color="blue")
    ax.plot(var_sans.index, var_sans.values, label="Sans DAS3H", color="gray")
    
    ax.set_xlabel("Semaine")
    ax.set_ylabel("Écart-type du PMR inter-élèves")
    ax.set_title("Dispersion du PMR — Avec vs Sans DAS3H")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_taux_reussite(simu_das3h, simu_sans, params):
    df_d = pd.DataFrame(simu_das3h.simulation_results)
    df_s = pd.DataFrame(simu_sans.simulation_results)
    
    taux_d = df_d.groupby("student")["correct"].mean()
    taux_s = df_s.groupby("student")["correct"].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(taux_d, bins=20, color="blue", alpha=0.7)
    ax1.set_xlabel("Taux de réussite")
    ax1.set_title(f"Avec DAS3H — std={taux_d.std():.3f}")
    ax1.axvline(x=0.5, color="red", linestyle="--")
    
    ax2.hist(taux_s, bins=20, color="gray", alpha=0.7)
    ax2.set_xlabel("Taux de réussite")
    ax2.set_title(f"Sans DAS3H — std={taux_s.std():.3f}")
    ax2.axvline(x=0.5, color="red", linestyle="--")
    
    plt.suptitle("Distribution du taux de réussite par élève", fontsize=14)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    #"Theta_tres(0.4)","Theta_tres_multiKC(0.4)","Mu_back_4","RandomH",
    df, q_matrix, kc_list = load_Model()
    kc_name_to_idx = {kc_list[i]: i for i in range(len(kc_list))}
    n_runs=3
    seeds=list(range(42,42+n_runs))
    n_ks=min(16, q_matrix.shape[1])
    time_review=1 #en heures
    exos=list(range(q_matrix.shape[0]))
    Graph_=Graph.CreateGraphSkills(df=df)
    Gkcs=Graph_.Creategraph()
    kcs_graph=Graph_.select_coherent_kcs(["Compter","Additionneer","Division",],kc_name_to_idx=kc_name_to_idx,max_kcs=16)
    all_runs_pmr={name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    all_runs_mastery={name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    all_runs_retention={name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    all_runs_global= {name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    for run, seed in enumerate(seeds):
        print(f"Run {run+1}/{n_runs} with seed {seed}")
        np.random.seed(seed)
        students = np.random.choice(df["user_id"].unique(), size=100, replace=False)
        kcs = list(np.random.choice(q_matrix.shape[1], size=n_ks, replace=False))
        params = {
            "alpha_s": {s: np.random.normal(0, 1) for s in students},
            "delta_j": {e: np.random.normal(1, 1) for e in exos},
            "beta_j":  {kc: np.random.normal(-1, 1) for kc in kcs},
            "theta_wins":     {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in kcs},
            "theta_attempts": {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in kcs},
        }
        
        heuristics = {
            "Mu_back_4": MuBackH.MuBackH(mu=4, kc_list=kcs,Graph=Gkcs),
            "Theta_tres(0.4)": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
            "Theta_tres_multiKC(0.4)": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4,multi_kc=True),
            "RandomH": RandomH.RandomH(kc_list=kcs),
            "Noreview": Noreview.Noreview(),
            
        }
        for name, heuristic in heuristics.items():
            print(f"Testing heuristic: {name}")
            simu_das3h = SimuH.SimulationH(
                    students=students, exos=exos, kcs=kcs, data=df, qmat=q_matrix,
                    heuristic=heuristic, history=False,
                    weeks_to_simulate=16, T_max_review_min=60*time_review, t0=0, kc_list=list(Gkcs.nodes))
            weekly_results,weekly_mastery, retention_pmr, global_pmr = simu_das3h.simulate_choffin(params)
            all_runs_retention[name].append(retention_pmr)
            all_runs_global[name].append(global_pmr)
           
    aggregated_retention = aggregate_runs(all_runs_retention)
    aggregated_global= aggregate_runs(all_runs_global)
    
    plot_aggregated(aggregated_global, "PMR moyen", 
                f"PMR global — Apprentissage + Rétention \n Protocle : 1h /week \n N° KCs = {n_ks} \n Nb_item =15 \n with Das3h ", 
                n_runs=n_runs, xlabel="Weeks")

print("Simulation completed for all heuristics.")