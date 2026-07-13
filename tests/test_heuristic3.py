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
import json 
import matplotlib.pyplot as plt
NAME_FOLDER="Mathiadata3" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 35717
FILE_PATH_JSON = "/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata/Kcs_Dependencies/KCs.json"

def load_data_js(file_path_json=FILE_PATH_JSON):
    with open(file_path_json, "r", encoding="utf-8") as f:
        return json.load(f)
def build_heuristics(all_kcs, data_js, kc_list):
    """Construit le dictionnaire des heuristiques à comparer."""
    return {
        "MuBack(μ=4)":   MuBackH.MuBackH(mu=4, kc_list=all_kcs, Graph=None),
        "ThetaThr(θ=0.4)": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
        "ZPD_window":    ZPDWd.ZPD_window(z1=0.2, z2=0.75),
        "ZPD_kcs":       ZPDH.ZPD_KCS(datajs=data_js, kclist=kc_list, z1=0.2, z2=0.75),
        "Random":        RandomH.RandomH(kc_list=all_kcs),
        "NoReview":      Noreview.Noreview(),
    }


def run_protocol(protocol, df, q_matrix, kc_list, data_js,
                 n_runs=1, n_ks=16, time_review=1.0, r=3,
                 n_students=100, verbose_idx=5):
   
    seeds = list(range(42, 42 + n_runs))
    n_ks = min(n_ks, q_matrix.shape[1])
    exos = list(range(q_matrix.shape[0]))
    kc_name_to_idx = {kc_list[i]: i for i in range(len(kc_list))}

    heuristic_names = ["MuBack(μ=4)", "ThetaThr(θ=0.4)",
                       "ZPD_window", "ZPD_kcs", "Random", "NoReview"]
    all_runs_global = {name: [] for name in heuristic_names}
    all_runs_simulation_results = {}
    all_runs_pmr_history = {}

    for run, seed in enumerate(seeds):
        print(f"[{protocol}] Run {run + 1}/{n_runs} (seed={seed})")
        np.random.seed(seed)

        students = np.random.choice(df["user_id"].unique(),
                                    size=n_students, replace=False)
        kcs = list(np.random.choice(q_matrix.shape[1], size=n_ks, replace=False))
        all_kcs = list(range(q_matrix.shape[1]))

        params = {
            "alpha_s": {s: np.random.normal(0, 1) for s in students},
            "delta_j": {e: np.random.normal(1, 1) for e in exos},
            "beta_j":  {kc: np.random.normal(-1, 1) for kc in all_kcs},
            "theta_wins":     {kc: [np.random.uniform(0, 2) for _ in range(5)]
                               for kc in all_kcs},
            "theta_attempts": {kc: [np.random.uniform(0, 2) for _ in range(5)]
                               for kc in all_kcs},
        }

        heuristics = build_heuristics(all_kcs, data_js, kc_list)

        for name, heuristic in heuristics.items():
            print(f"  → heuristique : {name}")
            simu = SimuH.SimulationH(
                students=students, exos=exos, kcs=kcs, data=df, qmat=q_matrix,
                heuristic=heuristic, history=False,
                weeks_to_simulate=16, T_max_review_min=60 * time_review,
                t0=0, kc_list=kcs,
            )

            if protocol == "choffin":
                _, _, retention_pmr, global_pmr = simu.simulate_choffin(params, r=r)
                pmr_history = None
            else:  # protocol == "time"
                (_, _, retention_pmr, global_pmr,
                 pmr_history) = simu.simulate(params, verbose_student=students[verbose_idx])

                # Timeline seulement pour les heuristiques ZPD en protocole "time"
                """if name in ("ZPD_kcs", "ZPD_window"):
                    plot_timeline(
                        simu.detailed_log,
                        kc_idx_to_name={i: kc_list[i] for i in range(len(kc_list))},
                        heuristic_name=name,
                        params=params,
                        student=students[verbose_idx],
                        items_per_kc=simu.items_per_kc,
                        curriculum_kcs=set(kcs),
                    )"""

            all_runs_global[name].append(global_pmr)
            all_runs_simulation_results[name] = simu.simulation_results
            if pmr_history is not None:
                all_runs_pmr_history[name] = pmr_history

    aggregated_global = aggregate_runs(all_runs_global)
    return aggregated_global, all_runs_simulation_results, all_runs_pmr_history


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
def test1():
    df, q_matrix, kc_list = load_Model()
    data_js = load_data_js(FILE_PATH_JSON)

    n_runs = 5
    n_ks = 16

    # --- Protocole 1 : Choffin (r révisions par semaine) ---
    r = 3
    agg_choffin, _, _ = run_protocol(
        protocol="choffin", df=df, q_matrix=q_matrix, kc_list=kc_list,
        data_js=data_js, n_runs=n_runs, n_ks=n_ks, r=r,
    )
    plot_aggregated(
        agg_choffin, "PMR moyen",
        f"PMR global — Protocole Choffin (r={r} révisions/sem.)\n"
        f"N° KCs = {n_ks}",
        n_runs=n_runs, xlabel="Semaines",
    )

    # --- Protocole 2 : 1h de révision continue par semaine ---
    time_review = 1.0  # en heures
    agg_time, _, _ = run_protocol(
        protocol="time", df=df, q_matrix=q_matrix, kc_list=kc_list,
        data_js=data_js, n_runs=n_runs, n_ks=n_ks, time_review=time_review,
    )
    plot_aggregated(
        agg_time, "PMR moyen",
        f"PMR global — Protocole {int(time_review * 60)} min/sem.\n"
        f"N° KCs = {n_ks}",
        n_runs=n_runs, xlabel="Semaines",
    )

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

def plot_timeline(detailed_log, kc_idx_to_name=None, heuristic_name="",
                  params=None, student=None, items_per_kc=None, curriculum_kcs=None):
    
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


if __name__ == "__main__":

    time_execute=1
    if time_execute==1: test1()
   
    print("!!!!!!!!!!!!!!!Done!!!!!!!!!!!!!!!!!")