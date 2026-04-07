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

NAME_FOLDER="Mathiadata" #algebra =574,item 1084
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
    plt.axvline(20, color='gray', linestyle='--', label="Fin de l'apprentissage")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} ({n_runs} runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



if __name__ == "__main__":
    #"Theta_tres(0.4)","Theta_tres_multiKC(0.4)","Mu_back_4","RandomH",
    df, q_matrix, kc_list = load_Model()
    n_runs=5
    seeds=list(range(42,42+n_runs))
    n_ks=min(16, q_matrix.shape[1])
    time_review=1 #en heures
    exos=list(range(q_matrix.shape[0]))
    all_runs_pmr={name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    all_runs_mastery={name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    all_runs_retention={name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
    all_runs_global = {name: [] for name in ["Noreview", "RandomH","Mu_back_4","Theta_tres(0.4)","Theta_tres_multiKC(0.4)"]}
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
            "Theta_tres(0.4)": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
            "Theta_tres_multiKC(0.4)": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4,multi_kc=True),
            "Mu_back_4": MuBackH.MuBackH(mu=4, kc_list=kcs),
            "RandomH": RandomH.RandomH(kc_list=kcs),
            "Noreview": Noreview.Noreview()
        }
        for name, heuristic in heuristics.items():
            print(f"Testing heuristic: {name}")
            simu = SimuH.SimulationH(
                students=students, exos=exos, kcs=kcs, data=df, qmat=q_matrix,
                heuristic=heuristic, history=False,
                weeks_to_simulate=20, T_max_review_min=60*time_review, t0=0
            )
            weekly_results,weekly_mastery, retention_pmr, global_pmr = simu.simulate_choffin(params)
            all_runs_pmr[name].append(weekly_results)
            all_runs_mastery[name].append(weekly_mastery)
            all_runs_retention[name].append(retention_pmr)
            all_runs_global[name].append(global_pmr)
    #aggregated_pmr = aggregate_runs(all_runs_pmr)
    #aggregated_mastery = aggregate_runs(all_runs_mastery)
    aggregated_retention = aggregate_runs(all_runs_retention)
    #plot_aggregated(aggregated_pmr, "PMR moyen", "Comparaison des heuristiques — PMR moyen par semaine", n_runs=n_runs)
    #plot_aggregated(aggregated_mastery, "% KCs maîtrisés (PMR ≥ 0.7)", "Comparaison des heuristiques — Taux de maîtrise par semaine", n_runs=n_runs)
    #plot_aggregated(aggregated_retention, "PMR moyen", "Comparaison des heuristiques — PMR de rétention", n_runs=n_runs, xlabel="Jours après la fin de l'apprentissage")
    aggregated_global = aggregate_runs(all_runs_global)
    plot_aggregated(aggregated_global, "PMR moyen", 
                f"PMR global — Apprentissage + Rétention \n Protocle : réviser 1h/week   \n N° KCs = {n_ks}", 
                n_runs=n_runs, xlabel="Weeks")

print("Simulation completed for all heuristics.")