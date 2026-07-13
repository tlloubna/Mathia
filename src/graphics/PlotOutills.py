import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np



HEURISTIC_COLORS = {
    "ThetaThr_0.4": plt.cm.Set2(0),
    "ZPD_window":      plt.cm.Set2(1),
    "ZPD_kcs":       plt.cm.Set2(2),
    "MuBack_mu4":   plt.cm.Set2(3),
    "Random":       plt.cm.Set2(4),
    "NoReview":     plt.cm.Set2(5),
}
def get_kc_colors(all_kcs):
    
    cmap = plt.cm.tab10
    kcs_sorted = sorted(all_kcs)
    return {kc: cmap(i % 10) for i, kc in enumerate(kcs_sorted)}

def plot_Q_matrix( Q=None, max_items=200):
    Q_small = Q[:max_items, :]
    plt.figure(figsize=(12, 8))
    sns.heatmap(Q_small, cmap="Greys", cbar=False)
    plt.xlabel("Skills (KC)")
    plt.ylabel("Items")
    plt.title(f"Matrix Q (display limited to {max_items} items)")
    plt.show()

    
def plot_performances(perf_df, students, metric="pmr_final_moyen"):
    for std in students:
        sub = perf_df[perf_df["élève"] == std]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sub["heuristique"], sub[metric],
            color=plt.cm.Set2(np.arange(len(sub))))
        ax.set_ylabel(metric)
        ax.set_title(f"Performance ({metric}) — Élève {std}")
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(sub[metric].values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()


def plot_efficacite(perf_df, students):
    """Croise effort (révisions) et résultat (gain moyen) par élève."""
    for std in students:
        sub = perf_df[perf_df["élève"] == std]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sub["total_révisions"], sub["gain_moyen"], s=120,
                color=plt.cm.Set2(np.arange(len(sub))), edgecolors="black",
                zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["heuristique"],
                        (r["total_révisions"], r["gain_moyen"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.set_xlabel("Total révisions ")
        ax.set_ylabel("Gain moyen de PMR (résultat)")
        ax.set_title(f"Efficacité : résultat vs Total révisions— Élève {std}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_choices_heatmap(df_table, heuristic_names, student, n_review_weeks=4):
    fig, axes = plt.subplots(1, len(heuristic_names),
                            figsize=(5 * len(heuristic_names), 6),
                            sharey=True)
    if len(heuristic_names) == 1:
        axes = [axes]

    df_std = df_table[df_table["student"] == student].set_index("KC")

    for ax, name in zip(axes, heuristic_names):
        cols = [f"{name}_w{w}" for w in range(n_review_weeks)]
        matrix = df_std[cols]
        matrix.columns = [f"S{w}" for w in range(n_review_weeks)]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd",
                    cbar_kws={"label": "nb choix"}, ax=ax,
                    vmin=0)
        ax.set_title(name)
        ax.set_xlabel("Semaine de révision")

    axes[0].set_ylabel("KC")
    plt.suptitle(f"Choix par heuristique — Élève {student}", fontsize=14)
    plt.tight_layout()
    plt.show()


    
def plot_pmr_timeline(pmr_evolution, choices_per_heuristic, student,
                    heuristic_names, n_review_weeks, kc_colors,
                    threshold_mastery=0.7, threshold_low=0.4):
    n_heuristics = len(heuristic_names)
    fig, axes = plt.subplots(n_heuristics, 1,
                            figsize=(14, 4 * n_heuristics),
                            sharex=True, sharey=True)
    if n_heuristics == 1:
        axes = [axes]

    x_ticks = list(range(-1, n_review_weeks))
    x_labels = ["init"] + [f"S{w}" for w in range(n_review_weeks)]

    all_kcs = sorted(pmr_evolution[heuristic_names[0]][-1].keys())

    for ax, name in zip(axes, heuristic_names):
        for kc in all_kcs:
            y = [pmr_evolution[name][w][kc] for w in x_ticks]
            ax.plot(x_ticks, y, marker="o", color=kc_colors[kc],
                    label=f"KC {kc}", linewidth=2, markersize=6, alpha=0.8)

            for w in range(n_review_weeks):
                n_choices = choices_per_heuristic[name].get(w, []).count(kc)
                if n_choices > 0:
                    pmr_val = pmr_evolution[name][w][kc]
                    ax.scatter(w, pmr_val, color=kc_colors[kc], marker="*",
                            s=200 + 30 * n_choices, zorder=5,
                            edgecolors="black", linewidths=1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("PMR")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                fontsize=8, ncol=1)

    axes[-1].set_xlabel("Semaine de révision")
    plt.suptitle(f"Évolution du PMR — Élève {student}\n"
                f"★ = KC révisé cette semaine (taille proportionnelle au nb de révisions)",
                fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_hist_pratique_par_eleve(df_hist, kc_colors, mode="count"):
    counts = df_hist.groupby(["user_id", "KC"]).size().unstack(fill_value=0)

    if mode == "proportion":
        counts = counts.div(counts.sum(axis=1), axis=0)
        ylabel = "Proportion des pratiques"
    else:
        ylabel = "Nombre de pratiques"

    students = counts.index.tolist()
    kcs = counts.columns.tolist()
    n_students = len(students)
    n_kcs = len(kcs)

    x = np.arange(n_students)
    width = 0.8 / n_kcs

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, kc in enumerate(kcs):
        offsets = x - 0.4 + width * (i + 0.5)
        ax.bar(offsets, counts[kc].values, width,
            label=f"KC {kc}", color=kc_colors.get(kc, "#cccccc"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Élève {s}" for s in students])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pratique des compétences sur l'historique ({mode})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def reconstruct_choices_from_table(df_table, student, heuristic_names,
                                    n_review_weeks):
    df_std = df_table[df_table["student"] == student]
    choices = {name: {w: [] for w in range(n_review_weeks)}
               for name in heuristic_names}
    for _, row in df_std.iterrows():
        kc = int(row["KC"])
        for name in heuristic_names:
            for w in range(n_review_weeks):
                n = int(row[f"{name}_w{w}"])
                choices[name][w].extend([kc] * n)
    return choices

def plot_par_niveau(agg_niveau, metric="gain_moyen"):
    pivot = agg_niveau[metric].unstack("heuristique")
   
    colors = [HEURISTIC_COLORS.get(h, "#cccccc") for h in pivot.columns]
    ax = pivot.plot(kind="bar", figsize=(11, 6),
                    color=colors, edgecolor="black")
    ax.set_ylabel(metric)
    ax.set_xlabel("Tranche de niveau initial")
    ax.set_title(f"{metric} par niveau initial et par heuristique")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.xticks(rotation=0)
    plt.legend(title="heuristique", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_global_comparison(agg, metric="gain_moyen", std_col="gain_std"):
    agg_sorted = agg.sort_values(metric, ascending=False)
    # couleurs dans l'ordre trié des heuristiques
    colors = [HEURISTIC_COLORS.get(h, "#cccccc") for h in agg_sorted.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(agg_sorted.index, agg_sorted[metric],
           yerr=agg_sorted.get(std_col), capsize=5,
           color=colors, edgecolor="black")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparaison globale ({metric}) — moyenne ± écart-type")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_global_violin(full, metric="gain_moyen"):
    ordre = (full.groupby("heuristique")[metric]
                 .median().sort_values(ascending=False).index.tolist())
    # palette = dict nom->couleur ; seaborn l'applique par nom, pas par position
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=full, x="heuristique", y=metric, order=ordre,
                   hue="heuristique", legend=False,
                   palette=HEURISTIC_COLORS, cut=0, inner="box", ax=ax)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Distribution de {metric} par heuristique "
                 f"({len(full)} observations)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_timing(df_timing, heuristic_colors=None):
    agg = df_timing.groupby("heuristique")["temps_moyen_ms"].agg(["mean", "std"])
    agg = agg.sort_values("mean")
    colors = ([heuristic_colors.get(h, "#cccccc") for h in agg.index]
              if heuristic_colors else None)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(agg.index, agg["mean"], yerr=agg["std"], capsize=5,
           color=colors, edgecolor="black")
    ax.set_ylabel("Temps moyen par décision (ms)")
    ax.set_title("Coût de calcul par heuristique (moyenne ± écart-type sur les élèves)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

def plot_timing_par_eleve(df_timing):
    pivot = df_timing.pivot(index="student", columns="heuristique",
                            values="temps_moyen_ms")
    pivot.plot(kind="bar", figsize=(12, 6), edgecolor="black")
    plt.ylabel("Temps moyen par décision (ms)")
    plt.title("Temps de décision par élève et par heuristique")
    plt.xticks(rotation=0)
    plt.legend(title="heuristique", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()