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
import src.Process.Heuristics.ZPD_KCs as ZPDH
import src.Process.Heuristics.ZPD_Pro as ZPDProp
import src.Process.Heuristics.ZPD_Wds as ZPDWd
NAME_FOLDER="Mathiadata3" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 35717
FILE_PATH_JSON = "/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata/Kcs_Dependencies/KCs.json"
PROFILES = {
    "maitrise":       {"n": 9, "window": (0.5, 1.0), "success_rate": 0.90},
    "en_cours":       {"n": 4, "window": (0.6, 1.0), "success_rate": 0.50},
    "ancien_oublie":  {"n": 5, "window": (0.0, 0.3), "success_rate": 0.70},
    "jamais_vu":      {"n": 0, "window": (0.0, 0.0), "success_rate": 0.0},
}
import matplotlib.pyplot as plt
"""def load_Model():
    # Load the model
    df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    kc_list=np.load(os.path.join(DATA_FOLDER,f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)["kc_list"]
    
    return df,q_matrix,kc_list"""

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


def plot_timeline(detailed_log, kc_idx_to_name=None, heuristic_name="",
                  params=None, student=None, items_per_kc=None, curriculum_kcs=None):
    """
    Timeline des décisions pour un élève.
    - Lignes pleines : KCs du curriculum
    - Lignes pointillées : KCs enfants (hors curriculum, explorés par ZPD)
    - ★ : révision d'un KC curriculum
    - ◆ : révision d'un KC enfant
    - Bande verte : zone ZPD [0.2, 0.7]
    """
    if not detailed_log:
        print("detailed_log vide, rien à tracer.")
        return

    curriculum_kcs = set(curriculum_kcs) if curriculum_kcs else set()

    # --- Collecter tous les KCs vus dans les logs ---
    all_kcs = set()
    for entry in detailed_log:
        all_kcs.update(entry["pmr_after"].keys())
    all_kcs = sorted(all_kcs)

    # Séparer curriculum vs enfants
    kcs_curriculum = [kc for kc in all_kcs if kc in curriculum_kcs]
    kcs_children = [kc for kc in all_kcs if kc not in curriculum_kcs]

    # --- Séries temporelles ---
    pmr_series = {kc: [] for kc in all_kcs}
    x_axis = []
    review_markers = []  # (x, kc, pmr, is_child)

    for i, entry in enumerate(detailed_log):
        x_axis.append(i)
        for kc in all_kcs:
            pmr_series[kc].append(entry["pmr_after"].get(kc, np.nan))
        for kc in entry["kcs_reviewed"]:
            is_child = kc in entry.get("kcs_are_children", [])
            pmr_val = entry["pmr_after"].get(kc, np.nan)
            review_markers.append((i, kc, pmr_val, is_child))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(18, 8))
    cmap = plt.cm.tab20
    colors = {kc: cmap(j % 20) for j, kc in enumerate(all_kcs)}
    def kc_label(kc):
        name = kc_idx_to_name.get(kc, str(kc)) if kc_idx_to_name else str(kc)
        if kc in curriculum_kcs:
            return name
        return f"{name} (enfant)"
    for kc in kcs_curriculum:
        ax.plot(x_axis, pmr_series[kc], color=colors[kc], alpha=0.7,
                linewidth=1.5, label=kc_label(kc))

    for kc in kcs_children:
        ax.plot(x_axis, pmr_series[kc], color=colors[kc], alpha=0.5,
                linewidth=1.2, linestyle="--", label=kc_label(kc))


    for (x, kc, pmr_val, is_child) in review_markers:
        if np.isnan(pmr_val):
            continue
        if is_child:
            ax.scatter(x, pmr_val, color=colors[kc], marker="D", s=80,
                       zorder=5, edgecolors="black", linewidths=0.5)
        else:
            ax.scatter(x, pmr_val, color=colors[kc], marker="*", s=120,
                       zorder=5, edgecolors="black", linewidths=0.5)

    prev_week = detailed_log[0]["week"]
    tick_positions = [0]
    tick_labels = [f"Sem. {prev_week}"]
    for i, entry in enumerate(detailed_log):
        if entry["week"] != prev_week:
            ax.axvline(i, color="gray", linestyle="--", alpha=0.4)
            tick_positions.append(i)
            tick_labels.append(f"Sem. {entry['week']}")
            prev_week = entry["week"]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Itérations (par semaine)")
    ax.set_ylabel("PMR")
    ax.set_title(f"Timeline — Élève {student} — {heuristic_name}\n"
                 f"★ = révision KC curriculum  ◆ = révision KC enfant")

    # Légende
    if len(all_kcs) <= 25:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                  ncol=1 if len(all_kcs) <= 15 else 2)
    else:
        ax.text(1.02, 0.5,
                f"{len(kcs_curriculum)} KCs curriculum\n{len(kcs_children)} KCs enfants\n(légende masquée)",
                transform=ax.transAxes, fontsize=9, va="center")

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def load_Model():
    df = pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    kc_list = np.load(os.path.join(DATA_FOLDER, f"history_metadata_{N_STUDENTS}std.npz"), 
                       allow_pickle=True)["kc_list"]
    kc_to_remove = "Être capable de privilégier les produits meilleurs pour sa santé, l'environnement et la vie locale"
    idx = np.where(kc_list == kc_to_remove)[0]
    if len(idx) > 0:
        idx = idx[0]
        print(f"Suppression du KC [{idx}]: {kc_to_remove}")
        q_matrix = np.delete(q_matrix, idx, axis=1)
        kc_list = np.delete(kc_list, idx)
        mask = df["KC"] != kc_to_remove
        df = df[mask].reset_index(drop=True)
        print(f"  Q-matrix: {q_matrix.shape}, kc_list: {len(kc_list)}, df: {len(df)}")
    
    return df, q_matrix, kc_list

def test1():
    #"Theta_tres(0.4)","Theta_tres_multiKC(0.4)","Mu_back_4","RandomH",
    df, q_matrix, kc_list = load_Model()
    kc_name_to_idx = {kc_list[i]: i for i in range(len(kc_list))}
    n_runs=1
    seeds=list(range(42,42+n_runs))
    n_ks=min(16, q_matrix.shape[1])
    time_list=[0.35]
    
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
                "MuBack(μ=4)":     MuBackH.MuBackH(mu=1, kc_list=all_kcs, Graph=None),
                "ThetaThr(0.4)":   Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
                "ZPD_window":   ZPDWd.ZPD_window(z1=0.2, z2=0.7),
                "ZPD_kcs":          ZPDH.ZPD_KCS(datajs=data_js, kclist=kc_list, z1=0.2, z2=0.75),
                "Random":          RandomH.RandomH(kc_list=all_kcs),
                "NoReview":        Noreview.Noreview(),
            }
            for name, heuristic in heuristics.items():
                print(f"Testing heuristic: {name}")
                simu_das3h = SimuH.SimulationH(
                        students=students, exos=exos, kcs=kcs, data=df, qmat=q_matrix,
                        heuristic=heuristic, history=False,
                        weeks_to_simulate=16, T_max_review_min=60*time_review, t0=0, kc_list=kcs)
                weekly_results,weekly_mastery, retention_pmr, global_pmr,pmr_history= simu_das3h.simulate(params,verbose_student=students[5])
                if name == "ZPD" or name == "Révision ciblée":
                        plot_timeline(
                            simu_das3h.detailed_log,
                            kc_idx_to_name={i: kc_list[i] for i in range(len(kc_list))},
                            heuristic_name=name,
                            params=params,
                            student=students[5],
                            items_per_kc=simu_das3h.items_per_kc,
                            curriculum_kcs=set(kcs),  # les KCs du curriculum
                        )
                all_runs_retention[name].append(retention_pmr)
                all_runs_global[name].append(global_pmr)
                all_runs_simulation_results[name] = simu_das3h.simulation_results
                all_runs_pmr_history[name] = pmr_history
        aggregated_retention = aggregate_runs(all_runs_retention)
        aggregated_global= aggregate_runs(all_runs_global)
        student_test = students[5]  # premier élève du run

        """all_gini = {}
        all_shannon = {}
        for name, sim_results in all_runs_simulation_results.items():
            if name != "Sans révision":
                plotHeatMapPMR(sim_results,all_runs_pmr_history[name] ,studentIndex=student_test,
                   kc_idx_to_name=None, heuristic_name=name)
                weeks, means, stds = ComputeGini_all(sim_results, all_runs_pmr_history[name], students)
                all_gini[name] = (weeks, means, stds)
                weeks, means, stds = ComputeShannon_all(sim_results, all_runs_pmr_history[name], students)
                all_shannon[name] = (weeks, means, stds)

        plot_all_diversity_avg(all_gini, all_shannon)"""
        plot_aggregated(aggregated_global, "PMR moyen", 
                    f"PMR global — Apprentissage + Rétention \n Protocle : {time_review*60} min/week \n N° KCs = {n_ks} ", 
                    n_runs=n_runs, xlabel="Weeks")

def _pick_item_for_kc(kc, qmatrix, rng):
    items = np.where(qmatrix[:, kc] == 1)[0]
    if len(items) == 0:
        return None
    return int(rng.choice(items))


def _finalize_history(rows):
    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "KC", "timestamp",
                                      "correct", "inter_id"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["inter_id"] = np.arange(len(df))
    return df

def CreateHistoryStudent_Scenario(students, qmatrix, profile_per_kc,n_days=10, t0=0, seed=42):
    rng = np.random.default_rng(seed)
    total_seconds = n_days * 24 * 3600
    rows = []
    for std in students:
        for kc, profile_name in profile_per_kc.items():
            if profile_name not in PROFILES:
                raise ValueError(f"Profil inconnu '{profile_name}' pour KC {kc}. "
                                 f"Choix : {list(PROFILES.keys())}")
            prof = PROFILES[profile_name]
            n_inter = prof["n"]
            if n_inter == 0:
                continue  
            t_start = t0 + prof["window"][0] * total_seconds
            t_end   = t0 + prof["window"][1] * total_seconds
            timestamps = np.sort(rng.uniform(t_start, t_end, size=n_inter))
            for ts in timestamps:
                item = _pick_item_for_kc(kc, qmatrix, rng)
                if item is None:
                    continue
                correct = int(rng.random() < prof["success_rate"])
                rows.append({
                    "user_id": std,
                    "item_id": item,
                    "KC": int(kc),
                    "timestamp": float(ts),
                    "correct": correct,
                    "inter_id": -1, 
                })
    return _finalize_history(rows)
def generate_qmatrix_controlled(n_kcs, items_per_kc=3):
    n_items = n_kcs * items_per_kc
    qmat = np.zeros((n_items, n_kcs), dtype=int)
    for kc in range(n_kcs):
        for k in range(items_per_kc):
            item_idx = kc * items_per_kc + k
            qmat[item_idx, kc] = 1
    return qmat



from utils.this_queue import OurQueue


def history_to_queues(history_df, student):
    df_std = history_df[history_df["user_id"] == student].sort_values("timestamp")
    queues = {}
    kcs_introduced = []
    for _, row in df_std.iterrows():
        kc = int(row["KC"])
        ts = float(row["timestamp"])
        correct = int(row["correct"])
        if kc not in queues:
            queues[kc] = {"wins": OurQueue(), "attempts": OurQueue()}
            kcs_introduced.append(kc)
        queues[kc]["attempts"].push(ts)
        if correct == 1:
            queues[kc]["wins"].push(ts)
    return queues, kcs_introduced
def history_to_last_review(history_df, student, seconds_per_week=7*24*3600):
    df_std = history_df[history_df["user_id"] == student]
    last_review = {}
    for kc in df_std["KC"].unique():
        df_kc = df_std[df_std["KC"] == kc]
        last_ts = df_kc["timestamp"].max()
        last_review[int(kc)] = int(last_ts // seconds_per_week)
    return last_review

import copy


def compute_pmr_for_kc(kc, params, queues, t_eval, alpha_s, items_per_kc):
    items = items_per_kc.get(kc, [])
    if len(items) == 0:
        delta_j = -1
    else:
        delta_j = params["delta_j"][int(items[0])]
    
    beta = params["beta_j"].get(kc, 0)
    
    if kc in queues:
        cw = queues[kc]["wins"].get_counters(t_eval)
        ca = queues[kc]["attempts"].get_counters(t_eval)
    else:
        cw = [0] * 5
        ca = [0] * 5
    
    h = sum(
        params["theta_wins"][kc][i] * np.log(1 + cw[i]) +
        params["theta_attempts"][kc][i] * np.log(1 + ca[i])
        for i in range(5)
    )
    logit = alpha_s - delta_j + beta + h
    return 1 / (1 + np.exp(-logit))


def classify_mastery(pmr):
    """Étiquette lisible pour le PMR."""
    if pmr >= 0.7:
        return "maîtrisé"
    elif pmr >= 0.4:
        return "en cours"
    elif pmr > 0:
        return "fragile"
    else:
        return "jamais vu"


def build_comparison_table(df_hist, qmat, params, heuristics, students,
                            t_eval, current_week, items_per_kc,
                            all_kcs, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for std in students:
        queues_init, kcs_init = history_to_queues(df_hist, student=std)
        last_review_init = history_to_last_review(df_hist, student=std)
        alpha_s = params["alpha_s"][std]
        pmr_per_kc = {
            kc: compute_pmr_for_kc(kc, params, queues_init, t_eval,
                                    alpha_s, items_per_kc)
            for kc in all_kcs
        }
        choices_per_heuristic = {}
        for name, heuristic in heuristics.items():
            queues_copy = copy.deepcopy(queues_init)
            if hasattr(heuristic, "reset"):
                heuristic.reset()
            if hasattr(heuristic, "last_review"):
                heuristic.last_review = dict(last_review_init)
            np.random.seed(seed)
            try:
                item, kcs_chosen = heuristic.HeuristicTochooseItemfromQ(
                    week=current_week,
                    kcs_introduced=kcs_init if kcs_init else all_kcs,
                    q_mat_=qmat,
                    student=std,
                    queues=queues_copy,
                    params=params,
                    t_current=t_eval,
                    items_per_kc=items_per_kc,
                    dictPkcs=pmr_per_kc,
                    kc_idx_to_name=None,
                    kc_name_to_idx=None,
                )
            except Exception as e:
                print(f"[!] {name} a échoué pour élève {std}: {e}")
                kcs_chosen = []
            
            choices_per_heuristic[name] = set(int(k) for k in (kcs_chosen or []))
        for kc in all_kcs:
            row = {"student": std, "KC": kc}
            for name in heuristics.keys():
                row[name] = 1 if kc in choices_per_heuristic[name] else 0
            row["pmr"] = round(pmr_per_kc[kc], 3)
            row["niveau_maitrise"] = classify_mastery(pmr_per_kc[kc])
            row["last_review_week"] = last_review_init.get(kc, None)
            rows.append(row)
    
    df_table = pd.DataFrame(rows)
    return df_table

def test2():
    n_kcs = 8
    items_per_kc_count = 3
    qmat = generate_qmatrix_controlled(n_kcs=n_kcs, items_per_kc=items_per_kc_count)
    
    profile = {
        0: "maitrise",
        1: "maitrise",
        2: "en_cours",
        3: "en_cours",
        4: "ancien_oublie",
        5: "ancien_oublie",
        6: "jamais_vu",
        7: "jamais_vu",
    }
    students = [101, 102]
    df_hist = CreateHistoryStudent_Scenario(
        students=students, qmatrix=qmat,
        profile_per_kc=profile, n_days=10, t0=0, seed=42,
    )
    np.random.seed(42)
    n_items = qmat.shape[0]
    all_kcs = list(range(qmat.shape[1]))
    
    params = {
        "alpha_s":        {s: np.random.normal(0, 1) for s in students},
        "delta_j":        {e: np.random.normal(1, 1) for e in range(n_items)},
        "beta_j":         {kc: np.random.normal(-1, 1) for kc in all_kcs},
        "theta_wins":     {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in all_kcs},
        "theta_attempts": {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in all_kcs},
    }
    items_per_kc = {}
    for kc in all_kcs:
        items = np.where(qmat[:, kc] == 1)[0]
        if len(items) > 0:
            items_per_kc[kc] = items
    heuristics = {
        "MuBack(μ=4)":     MuBackH.MuBackH(mu=1, kc_list=all_kcs, Graph=None),
        "ThetaThr(0.4)":   Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
        "ZPD_window":   ZPDWd.ZPD_window(z1=0.2, z2=0.7),
         "ZPD_kcs":          ZPDH.ZPD_KCS(datajs=data_js, kclist=kc_list, z1=0.2, z2=0.75),
        "Random":          RandomH.RandomH(kc_list=all_kcs),
        "NoReview":        Noreview.Noreview(),
    }
    n_days = 10
    t_eval = n_days * 24 * 3600
    current_week = n_days // 7  
    df_table = build_comparison_table(
        df_hist=df_hist, qmat=qmat, params=params,
        heuristics=heuristics, students=students,
        t_eval=t_eval, current_week=current_week,
        items_per_kc=items_per_kc, all_kcs=all_kcs,
        seed=42,
    )
    
    print("\n=== HISTORIQUE ===")
    print(df_hist)
    print("\n=== TABLEAU COMPARATIF ===")
    print(df_table.to_string(index=False))
    
    return df_hist, df_table
if __name__ == "__main__":

    time_execute=1
    if time_execute==1: test1()
    else : 
        df_hist, df_table=test2()
        print(df_hist)
        print(df_table)
        print("!!!!!!!!!!!!!!!Done!!!!!!!!!!!!!!!!!")
print("Simulation completed for all heuristics.")