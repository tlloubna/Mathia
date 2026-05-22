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
import seaborn as sns
import src.Process.Heuristics.ZPD_KCs as ZPD_KCS
NAME_FOLDER="Mathiadata3" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 35717
FILE_PATH_JSON = "/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata/Kcs_Dependencies/KCs.json"

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
    plt.legend(fontsize=15)
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
    ax.legend(fontsize=15)
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



def ComputeGini(simulation_results, pmr_history, studentIndex):
    data = pd.DataFrame.from_dict(simulation_results)
    stddf = data[data["student"] == studentIndex].explode("kcs")
    weeks = sorted(pmr_history[studentIndex].keys())
    all_kcs = sorted(stddf["kcs"].unique())
    
    kc_to_row = {kc: i for i, kc in enumerate(all_kcs)}
    week_to_col = {w: j for j, w in enumerate(weeks)}
    rev_matrix = np.zeros((len(all_kcs), len(weeks)))
    for _, row in stddf.iterrows():
        if row["kcs"] in kc_to_row and row["week"] in week_to_col:
            rev_matrix[kc_to_row[row["kcs"]], week_to_col[row["week"]]] += 1
    
    Gini_list = []
    n = len(all_kcs)
    for w in range(rev_matrix.shape[1]):
        Sum = 0
        M = 0
        for i in range(n):
            M += rev_matrix[i][w]
            for j in range(n):
                Sum += np.abs(rev_matrix[i][w] - rev_matrix[j][w])
        E = Sum / n**2
        M /= n
        Gini = E / (2 * M) if M > 0 else 0
        Gini_list.append(Gini)
    
    return weeks, Gini_list


    



def plotHeatMapPMR(simulation_results, pmr_history, studentIndex, 
                   kc_idx_to_name=None, heuristic_name=""):
    data = pd.DataFrame.from_dict(simulation_results)
    stddf = data[data["student"] == studentIndex].explode("kcs")
    weeks = sorted(pmr_history[studentIndex].keys())
    all_kcs = sorted(stddf["kcs"].unique())
    pmr_matrix = np.zeros((len(all_kcs), len(weeks)))
    for j, w in enumerate(weeks):
        for i, kc in enumerate(all_kcs):
            pmr_matrix[i, j] = pmr_history[studentIndex][w].get(kc, 0)

    kc_to_row = {kc: i for i, kc in enumerate(all_kcs)}
    week_to_col = {w: j for j, w in enumerate(weeks)}
    rev_matrix = np.zeros_like(pmr_matrix)
    for _, row in stddf.iterrows():
        if row["kcs"] in kc_to_row and row["week"] in week_to_col:
            rev_matrix[kc_to_row[row["kcs"]], week_to_col[row["week"]]] += 1

    annot = np.empty_like(pmr_matrix, dtype=object)
    for i in range(pmr_matrix.shape[0]):
        for j in range(pmr_matrix.shape[1]):
            pmr_str = f"{pmr_matrix[i,j]:.2f}"
            rev = int(rev_matrix[i, j])
            if rev > 0:
                annot[i, j] = f"{pmr_str}\n({rev}x)"
            else:
                annot[i, j] = pmr_str

    if kc_idx_to_name:
        yticklabels = [kc_idx_to_name.get(kc, str(kc)) for kc in all_kcs]
    else:
        yticklabels = [str(kc) for kc in all_kcs]

    plt.figure(figsize=(14, max(5, len(all_kcs) * 0.5)))
    sns.heatmap(pmr_matrix, annot=annot, fmt="", cmap="YlOrRd",
                vmin=0, vmax=1,
                xticklabels=weeks, yticklabels=yticklabels,
                cbar_kws={"label": "PMR"})
    plt.xlabel("Semaine")
    plt.ylabel("KC")
    plt.title(f"PMR + Révisions — Élève {studentIndex} — {heuristic_name}")
    plt.tight_layout()
    plt.show()
def ComputeShannon(simulation_results, pmr_history, studentIndex):
    data = pd.DataFrame.from_dict(simulation_results)
    stddf = data[data["student"] == studentIndex].explode("kcs")
    weeks = sorted(pmr_history[studentIndex].keys())
    all_kcs = sorted(stddf["kcs"].unique())
    
    kc_to_row = {kc: i for i, kc in enumerate(all_kcs)}
    week_to_col = {w: j for j, w in enumerate(weeks)}
    rev_matrix = np.zeros((len(all_kcs), len(weeks)))
    for _, row in stddf.iterrows():
        if row["kcs"] in kc_to_row and row["week"] in week_to_col:
            rev_matrix[kc_to_row[row["kcs"]], week_to_col[row["week"]]] += 1
    
    shannon_list = []
    n = len(all_kcs)
    H_max = np.log(n) if n > 1 else 1  # entropie maximale (répartition uniforme)
    
    for w in range(rev_matrix.shape[1]):
        total = rev_matrix[:, w].sum()
        if total == 0:
            shannon_list.append(0)
            continue
        
        # Proportions
        p = rev_matrix[:, w] / total
        
        # Entropie de Shannon (ignorer les p=0 car 0*log(0) = 0)
        H = -np.sum(p[p > 0] * np.log(p[p > 0]))
        
        # Normaliser entre 0 et 1 (evenness / équitabilité de Pielou)
        J = H / H_max if H_max > 0 else 0
        
        shannon_list.append(J)
    
    return weeks, shannon_list

def ComputeShannon_all(simulation_results, pmr_history, students):
    all_shannon = []
    
    for student in students:
        if student not in pmr_history:
            continue
        weeks, shannon = ComputeShannon(simulation_results, pmr_history, student)
        if shannon:
            all_shannon.append(shannon)
    
    all_shannon = np.array(all_shannon)  # shape: (n_students, n_weeks)
    weeks = sorted(pmr_history[students[0]].keys())
    
    means = np.mean(all_shannon, axis=0)
    stds = np.std(all_shannon, axis=0)
    
    return weeks, means, stds

def ComputeGini_all(simulation_results, pmr_history, students):
    all_gini = []
    
    for student in students:
        if student not in pmr_history:
            continue
        weeks, gini = ComputeGini(simulation_results, pmr_history, student)
        if gini:
            all_gini.append(gini)
    
    all_gini = np.array(all_gini)
    weeks = sorted(pmr_history[students[0]].keys())
    
    means = np.mean(all_gini, axis=0)
    stds = np.std(all_gini, axis=0)
    
    return weeks, means, stds

def plot_all_diversity_avg(all_gini, all_shannon):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    for name, (weeks, means, stds) in all_gini.items():
        ax1.plot(weeks, means, label=name, linewidth=2)
        ax1.fill_between(weeks, means - stds, means + stds, alpha=0.2)
    ax1.set_xlabel("Semaine")
    ax1.set_ylabel("Indice de Gini")
    ax1.set_title("Concentration des révisions (Gini)\nMoyenne ± écart-type sur tous les élèves")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    for name, (weeks, means, stds) in all_shannon.items():
        ax2.plot(weeks, means, label=name, linewidth=2)
        ax2.fill_between(weeks, means - stds, means + stds, alpha=0.2)
    ax2.set_xlabel("Semaine")
    ax2.set_ylabel("Équitabilité de Pielou (J)")
    ax2.set_title("Diversité des révisions (Shannon normalisé)\nMoyenne ± écart-type sur tous les élèves")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
def plot_timeline(detailed_log, kc_idx_to_name=None, heuristic_name="",params=None,student=None,items_per_kc=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    chosen_kcs = sorted(set(entry["chosen_kc"] for entry in detailed_log))
    colors = plt.cm.tab20(np.linspace(0, 1, len(chosen_kcs)))
    
    for idx, kc in enumerate(chosen_kcs):
       
        pmrs_x = []
        pmrs_y = []
        if kc==79:
            print("stp")
        
        for i, entry in enumerate(detailed_log):
            if kc in entry["pmr_before"]:
                pmrs_x.append(i)
                pmrs_y.append(entry["pmr_before"][kc])
            else : 
                alpha_s = params["alpha_s"][student]
                beta = params["beta_j"].get(kc, 0)
                items = items_per_kc.get(kc, [])
                if len(items) == 0:
                    delta_j=-1
                else:
                    item=items[0]
                    delta_j = params["delta_j"][item]
                queues=entry["queues"]
                t_eval = entry["t_current"]
                if kc in queues:
                    cw = queues[kc]["wins"].get_counters(t_eval)
                    ca = queues[kc]["attempts"].get_counters(t_eval)
                else:
                    cw = [0] * 5
                    ca = [0] * 5
                h = sum(
                    params["theta_wins"].get(kc, [0]*5)[i] * np.log(1 + cw[i])
                    + params["theta_attempts"].get(kc, [0]*5)[i] * np.log(1 + ca[i])
                    for i in range(5)
                )
                logit = alpha_s - delta_j + beta + h
                p_=1/(1+np.exp(-logit))
                pmrs_x.append(i)
                pmrs_y.append(p_)
        name = kc_idx_to_name.get(kc, str(kc)) if kc_idx_to_name else str(kc)
        ax.plot(pmrs_x, pmrs_y, color=colors[idx], alpha=0.6, linewidth=1.5, label=name)
        
        for i, entry in enumerate(detailed_log):
            if entry["chosen_kc"] == kc:
                marker = "^" if entry["correct"] else "v"
                ax.scatter(i, entry["pmr_before"].get(kc, 0), marker=marker,
                          color=colors[idx], s=80, zorder=5, 
                          edgecolors="black", linewidth=0.5)
    
    weeks_seen = set()
    for i, entry in enumerate(detailed_log):
        if entry["week"] not in weeks_seen:
            ax.axvline(i, color="gray", linestyle="--", alpha=0.3)
            ax.text(i, 1.02, f"S{entry['week']}", fontsize=8,
                   color="gray", ha="center", transform=ax.get_xaxis_transform())
            weeks_seen.add(entry["week"])
    
    ax.set_xlabel("Itération")
    ax.set_ylabel("PMR")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Timeline des décisions — {heuristic_name}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    #"Theta_tres(0.4)","Theta_tres_multiKC(0.4)","Mu_back_4","RandomH",
    df, q_matrix, kc_list = load_Model()
    kc_name_to_idx = {kc_list[i]: i for i in range(len(kc_list))}
    n_runs=1
    seeds=list(range(42,42+n_runs))
    n_ks=min(40, q_matrix.shape[1])
    time_list=[0.45]
    
    #time_review=1 #en heures
    exos=list(range(q_matrix.shape[0]))
    #Graph_=Graph.CreateGraphSkills(df=df)
    #Gkcs=Graph_.Creategraph()
    #kcs_graph=Graph_.select_coherent_kcs(["Compter","Additionneer","Division",],kc_name_to_idx=kc_name_to_idx,max_kcs=16)
    for time_review in time_list:
        all_runs_pmr={name: [] for name in ["Sans révision", "Révision aléatoire","Révision à espacement fixe","Révision ciblée","ZPD"]}
        all_runs_mastery={name: [] for name in ["Sans révision", "Révision aléatoire","Révision à espacement fixe","Révision ciblée","ZPD"]}
        all_runs_retention={name: [] for name in ["Sans révision", "Révision aléatoire","Révision à espacement fixe","Révision ciblée","ZPD"]}
        all_runs_global= {name: [] for name in ["Sans révision", "Révision aléatoire","Révision à espacement fixe","Révision ciblée","ZPD"]}
        all_runs_simulation_results = {}
        all_runs_pmr_history = {}
        """items_per_kc = {}
        for kc in range(q_matrix.shape[1]):
            items = np.where(q_matrix[:, kc] == 1)[0]
            if len(items) > 0:
                items_per_kc[kc] = items"""
        for run, seed in enumerate(seeds):
            print(f"Run {run+1}/{n_runs} with seed {seed}")
            np.random.seed(seed)
            students = np.random.choice(df["user_id"].unique(), size=100, replace=False)
            kcs = list(np.random.choice(q_matrix.shape[1], size=n_ks, replace=False))
            all_kcs = list(range(q_matrix.shape[1]))

            params = {
                "alpha_s": {s: np.random.normal(0, 1) for s in students},
                "delta_j": {e: np.random.normal(1, 1) for e in exos},
                "beta_j":  {kc: np.random.normal(-1, 1) for kc in all_kcs},
                "theta_wins":     {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in all_kcs},
                "theta_attempts": {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in all_kcs},
            }
            
            heuristics = {
                "ZPD": ZPD_KCS.ZPD_KCS(pathfilejs=FILE_PATH_JSON, kclist=kc_list, z1=0.2, z2=0.7),
                "Révision à espacement fixe": MuBackH.MuBackH(mu=4, kc_list=kcs, Graph=None),
                "Révision ciblée": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
                #"Révision ciblée multiKcs": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4, multi_kc=True),
                "Révision aléatoire": RandomH.RandomH(kc_list=kcs),
                "Sans révision": Noreview.Noreview(),
                
            }
            for name, heuristic in heuristics.items():
                print(f"Testing heuristic: {name}")
                simu_das3h = SimuH.SimulationH(
                        students=students, exos=exos, kcs=kcs, data=df, qmat=q_matrix,
                        heuristic=heuristic, history=False,
                        weeks_to_simulate=16, T_max_review_min=60*time_review, t0=0, kc_list=kcs)
                weekly_results,weekly_mastery, retention_pmr, global_pmr,pmr_history= simu_das3h.simulate(params,verbose_student=students[5])
                """if name=="ZPD" or name=="Révision ciblée" :
                    plot_timeline(simu_das3h.detailed_log, kc_idx_to_name=None, heuristic_name=name,params=params,student=students[5],items_per_kc=items_per_kc)"""
                all_runs_retention[name].append(retention_pmr)
                all_runs_global[name].append(global_pmr)
                all_runs_simulation_results[name] = simu_das3h.simulation_results
                all_runs_pmr_history[name] = pmr_history
        aggregated_retention = aggregate_runs(all_runs_retention)
        aggregated_global= aggregate_runs(all_runs_global)
        student_test = students[5]  # premier élève du run

        all_gini = {}
        all_shannon = {}
        for name, sim_results in all_runs_simulation_results.items():
            if name != "Sans révision":
                weeks, means, stds = ComputeGini_all(sim_results, all_runs_pmr_history[name], students)
                all_gini[name] = (weeks, means, stds)
                weeks, means, stds = ComputeShannon_all(sim_results, all_runs_pmr_history[name], students)
                all_shannon[name] = (weeks, means, stds)

        plot_all_diversity_avg(all_gini, all_shannon)
               

        plot_aggregated(aggregated_global, "PMR moyen", 
                    f"PMR global — Apprentissage + Rétention \n Protocle : {time_review*60} min/week \n N° KCs = {n_ks} ", 
                    n_runs=n_runs, xlabel="Weeks")


print("Simulation completed for all heuristics.")