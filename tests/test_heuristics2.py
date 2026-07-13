# tests/test_heuristics2.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import src.Process.Heuristics.mu_back as MuBackH
import src.Process.Heuristics.Random as RandomH
import src.Process.Heuristics.no_review as Noreview
import src.Process.Heuristics.Theta_TresholdH as Theta_TresholdH
import src.Process.Heuristics.ZPD_KCs as ZPDH
import src.Process.Heuristics.ZPD_Pro as ZPDProp
import src.Process.Heuristics.ZPD_Wds as ZPDWd
from src.Process.Simulation.utils import (
    generate_qmatrix_controlled, generate_skill_tree_json,
    assign_coherent_profiles, CreateHistoryStudent_Scenario,
    compute_pmr_for_kc,generate_random_students
)
from src.Process.Simulation.Evaluator import Evaluator
from src.Analysis.SimAnalyzer import performances_par_eleve, print_performances
from src.graphics.PlotOutills import (
    get_kc_colors, plot_performances, plot_efficacite,
    plot_choices_heatmap, plot_pmr_timeline, plot_hist_pratique_par_eleve,
    reconstruct_choices_from_table,plot_global_comparison,plot_par_niveau,plot_global_violin,plot_timing
)
from src.Analysis.SimAnalyzer import resume_timing
from src.graphics.Plotgraphnewtorks import plot_dependency_graph
import pandas as pd
import time 




def RunforTWoStudent():
    structure = {
        "Nombres":      ["Addition", "Soustraction"],
        "Addition":     ["Addition simple", "Addition retenue"],
        "Soustraction": ["Soustraction simple"],
        "Géométrie":    ["Figures", "Angles"],
        "Figures":      ["Triangle"],
    }
    data_js, kc_list = generate_skill_tree_json(structure)
    n_kcs = len(kc_list)
    qmat = generate_qmatrix_controlled(n_kcs=n_kcs, items_per_kc=3)

    leaf_101 = {"Addition simple": "maitrise", "Addition retenue": "en_cours",
                "Soustraction simple": "ancien_oublie", "Triangle": "en_cours",
                "Angles": "ancien_oublie"}
    leaf_102 = {"Addition simple": "en_cours", "Addition retenue": "ancien_oublie",
                "Soustraction simple": "jamais_vu", "Triangle": "ancien_oublie",
                "Angles": "jamais_vu"}

    def names_to_idx(prof):
        return {idx: prof[name] for idx, name in enumerate(kc_list) if name in prof}

    profile_per_eleve = {
        101: names_to_idx(assign_coherent_profiles(structure, leaf_101)),
        102: names_to_idx(assign_coherent_profiles(structure, leaf_102)),
    }

    students = [101, 102]
    hist_n_days, n_review_weeks, t_max_minutes = 30, 10, 30
    alpha_par_eleve = {101: 0.5, 102: -1.5}

    df_hist = CreateHistoryStudent_Scenario(
        students=students, qmatrix=qmat, profile_per_kc=profile_per_eleve,
        n_days=hist_n_days, t0=0, seed=42,
    )

    np.random.seed(42)
    n_items = qmat.shape[0]
    all_kcs = list(range(n_kcs))
    kc_colors = get_kc_colors(all_kcs)

    plot_hist_pratique_par_eleve(df_hist, kc_colors, mode="count")

    params = {
        "alpha_s":       alpha_par_eleve,
        "delta_j":        {e: np.random.normal(1, 1) for e in range(n_items)},
        "beta_j":         {kc: np.random.normal(-1, 1) for kc in all_kcs},
        "theta_wins":     {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in all_kcs},
        "theta_attempts": {kc: [np.random.uniform(0, 2) for _ in range(5)] for kc in all_kcs},
    }
    items_per_kc = {kc: np.where(qmat[:, kc] == 1)[0] for kc in all_kcs}
    evaluator = Evaluator()
    for std in students:
        queues_init, _ = evaluator.history_to_queues(df_hist, student=std)
        t_review_start = hist_n_days * 24 * 3600
        pmr_initial = {
            kc: compute_pmr_for_kc(kc, params, queues_init, t_review_start,
                                   params["alpha_s"][std], items_per_kc)
            for kc in all_kcs
        }
        plot_dependency_graph(structure, kc_list, pmr_initial,
                              student=std, kc_colors=kc_colors, z1=0.2, z2=0.75)

    heuristics = {
            "ThetaThr_0.4": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
            "ZPD_window":   ZPDWd.ZPD_window(z1=0.2, z2=0.7),
            "ZPD_kcs":          ZPDH.ZPD_KCS(datajs=data_js, kclist=kc_list, z1=0.2, z2=0.75),
            "MuBack_mu4":   MuBackH.MuBackH(mu=4, kc_list=all_kcs, Graph=None),
            "Random":       RandomH.RandomH(kc_list=all_kcs), }
    df_table, pmr_evolution_all ,df_timing= evaluator.build_comparison_table_multiweek(
        df_hist=df_hist, qmat=qmat, params=params, heuristics=heuristics,
        students=students, items_per_kc=items_per_kc, all_kcs=all_kcs,
        n_review_weeks=n_review_weeks, t_max_minutes=t_max_minutes,
        hist_n_days=hist_n_days, seed=42, kcs_mode="all",
    )

    perf_df = performances_par_eleve(df_table, list(heuristics.keys()),
                                     students, n_review_weeks)
    print_performances(perf_df, students)
    plot_performances(perf_df, students, metric="pmr_final_moyen")
    plot_efficacite(perf_df, students)

    print("\n=== TEMPS DE CALCUL ===")
    print(resume_timing(df_timing).to_string())
    plot_timing(df_timing)
    names_heatmap = [n for n in heuristics if n != "NoReview"]
    names_timeline = list(heuristics.keys())
    """for std in students:
        plot_choices_heatmap(df_table, names_heatmap, std, n_review_weeks)
        choices = reconstruct_choices_from_table(df_table, std, names_timeline, n_review_weeks)
        plot_pmr_timeline(pmr_evolution_all[std], choices, std,
                          names_timeline, n_review_weeks, kc_colors)"""

    return df_hist, df_table, perf_df


def run_one_simulation(structure, kc_list, data_js, n_students, seed,
                       hist_n_days=30, n_review_weeks=10, t_max_minutes=30):
    rng = np.random.default_rng(seed)
    n_kcs = len(kc_list)
    qmat = generate_qmatrix_controlled(n_kcs=n_kcs, items_per_kc=3)
    all_kcs = list(range(n_kcs))
    n_items = qmat.shape[0]

    profile_per_eleve, alpha_par_eleve, students = generate_random_students(
        structure, kc_list, n_students, rng
    )

    df_hist = CreateHistoryStudent_Scenario(
        students=students, qmatrix=qmat, profile_per_kc=profile_per_eleve,
        n_days=hist_n_days, t0=0, seed=seed,
    )

    params = {
        "alpha_s":       alpha_par_eleve,
        "delta_j":        {e: float(rng.normal(1, 1)) for e in range(n_items)},
        "beta_j":         {kc: float(rng.normal(-1, 1)) for kc in all_kcs},
        "theta_wins":     {kc: [float(rng.uniform(0, 2)) for _ in range(5)] for kc in all_kcs},
        "theta_attempts": {kc: [float(rng.uniform(0, 2)) for _ in range(5)] for kc in all_kcs},
    }
    items_per_kc = {kc: np.where(qmat[:, kc] == 1)[0] for kc in all_kcs}

    heuristics = {
            "ThetaThr_0.4": Theta_TresholdH.ThetaTresholdH(theta_threshold=0.4),
            "ZPD_window":   ZPDWd.ZPD_window(z1=0.2, z2=0.7),
            "ZPD_kcs":          ZPDH.ZPD_KCS(datajs=data_js, kclist=kc_list, z1=0.2, z2=0.75),
            "MuBack_mu4":   MuBackH.MuBackH(mu=4, kc_list=all_kcs, Graph=None),
            "Random":       RandomH.RandomH(kc_list=all_kcs), }

    evaluator = Evaluator()
    df_table, _,df_timing = evaluator.build_comparison_table_multiweek(
        df_hist=df_hist, qmat=qmat, params=params, heuristics=heuristics,
        students=students, items_per_kc=items_per_kc, all_kcs=all_kcs,
        n_review_weeks=n_review_weeks, t_max_minutes=t_max_minutes,
        hist_n_days=hist_n_days, seed=seed, kcs_mode="all",
    )

    perf = performances_par_eleve(df_table, list(heuristics.keys()),
                                  students, n_review_weeks)
    perf["seed"] = seed
    df_timing["seed"] = seed           
    return perf,df_timing

def run_many_simulations(structure, kc_list, data_js,
                         n_students=50, n_simulations=20):
    all_perfs, all_timings = [], []
    for sim in range(n_simulations):
        perf,timing = run_one_simulation(structure, kc_list, data_js,
                                  n_students=n_students, seed=sim)
        all_perfs.append(perf)
        all_timings.append(timing)
        print(f"  simulation {sim+1}/{n_simulations} terminée")

    full = pd.concat(all_perfs, ignore_index=True)
    full_timing = pd.concat(all_timings, ignore_index=True)
    agg = full.groupby("heuristique").agg(
        pmr_final_moyen=("pmr_final_moyen", "mean"),
        pmr_final_std=("pmr_final_moyen", "std"),
        gain_moyen=("gain_moyen", "mean"),
        gain_std=("gain_moyen", "std"),
        gain_par_rev=("gain_par_révision", "mean"),
        revisions=("total_révisions", "mean"),
    ).round(4)
    return full, agg,full_timing
def agg_par_niveau(full, n_tranches=3):
    """Agrège les perfs par tranche de niveau initial et par heuristique."""
    labels = ["faible", "moyen", "fort"] if n_tranches == 3 else None
    full = full.copy()
    full["tranche"] = pd.qcut(full["niveau_initial"], q=n_tranches, labels=labels)
    agg = full.groupby(["tranche", "heuristique"], observed=True).agg(
        gain_moyen=("gain_moyen", "mean"),
        gain_std=("gain_moyen", "std"),
        pmr_final=("pmr_final_moyen", "mean"),
        n=("gain_moyen", "size"),
    ).round(4)
    return agg

def main_global(n_students=50, n_simulations=20):
    structure = {
        "Nombres":      ["Addition", "Soustraction"],
        "Addition":     ["Addition simple", "Addition retenue"],
        "Soustraction": ["Soustraction simple"],
        "Géométrie":    ["Figures", "Angles"],
        "Figures":      ["Triangle"],
    }
    data_js, kc_list = generate_skill_tree_json(structure)

    full, agg,full_timing = run_many_simulations(structure, kc_list, data_js,
                                 n_students=n_students, n_simulations=n_simulations)
    agg_niveau = agg_par_niveau(full, n_tranches=3)
    print(agg_niveau.to_string())
    plot_par_niveau(agg_niveau, "gain_moyen")
    print("\n", agg.to_string())
    plot_global_comparison(agg, "gain_moyen", "gain_std")
    plot_global_comparison(agg, "pmr_final_moyen", "pmr_final_std")
    plot_global_violin(full, metric="gain_moyen")
    print("\n=== TEMPS DE CALCUL (sur toutes les simulations) ===")
    print(resume_timing(full_timing).to_string())
    plot_timing(full_timing)
    return full, agg,full_timing


if __name__ == "__main__":
    #RunforTWoStudent()
    full, agg,full_timing= main_global(n_students=100, n_simulations=10)
    print("\n!!!!! Done !!!!!")