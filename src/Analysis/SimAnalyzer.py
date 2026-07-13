

import pandas as pd 

import matplotlib.pyplot as plt 


def print_heuristic_view(df_table, heuristic_name, n_review_weeks=4):
    cols = ["student", "KC", "niveau_initial", "pmr_initial",
            "last_review_initial"]
    cols += [f"{heuristic_name}_w{w}" for w in range(n_review_weeks)]
    cols += [f"{heuristic_name}_total", f"{heuristic_name}_pmr_final"]
    print(f"\n=== {heuristic_name} ===")
    print(df_table[cols].to_string(index=False))

def performances_par_eleve(df_table, heuristic_names, students,
                           n_review_weeks, seuil_maitrise=0.7):
    
    rows = []
    for std in students:
        df_std = df_table[df_table["student"] == std]
        n_kc = len(df_std)
        for name in heuristic_names:
            col_final = f"{name}_pmr_final"
            pmr_final_moyen = df_std[col_final].mean()
            n_maitrises = int((df_std[col_final] >= seuil_maitrise).sum())
            total_revisions = sum(
                int(df_std[f"{name}_w{w}"].sum()) for w in range(n_review_weeks)
            )
            gain_moyen = (df_std[col_final] - df_std["pmr_initial"]).mean()
            gain_total = (df_std[col_final] - df_std["pmr_initial"]).sum()
            gain_par_rev = gain_total / total_revisions if total_revisions else 0.0
            niveau_initial = df_std["pmr_initial"].mean() 
            rows.append({
                "élève": std,
                "heuristique": name,
                "niveau_initial": round(niveau_initial, 3), 
                "pmr_final_moyen": round(pmr_final_moyen, 3),
                "kc_maitrisés": f"{n_maitrises}/{n_kc}",
                "gain_moyen": round(gain_moyen, 3),
                "total_révisions": total_revisions,
                "gain_par_révision": round(gain_par_rev, 4),
            })
    return pd.DataFrame(rows)


def print_performances(perf_df, students):
    for std in students:
        print(f"\n{'='*70}")
        print(f"PERFORMANCES — Élève {std}")
        print('='*70)
        sub = perf_df[perf_df["élève"] == std].drop(columns=["élève"])
        print(sub.to_string(index=False))

def resume_timing(df_timing):
    """Temps moyen de décision par heuristique (moyenné sur les élèves)."""
    agg = df_timing.groupby("heuristique").agg(
        temps_moyen_ms=("temps_moyen_ms", "mean"),
        temps_std_ms=("temps_moyen_ms", "std"),
        n_decisions_moy=("n_decisions", "mean"),
    ).round(4).sort_values("temps_moyen_ms")
    return agg

