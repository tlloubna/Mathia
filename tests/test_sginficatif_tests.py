import os
import sys

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations


N_STUDENTS = 1146
FOLDER ="bridge_algebra06/results"
DATA_FOLDER=os.path.join("data",FOLDER)
DF = pd.read_csv(os.path.join(DATA_FOLDER, "cv_folds.csv"))
# ordre d'affichage (adaptez si un modèle manque)
MODEL_ORDER = ["DAS3H+Bayes", "DAS3H user/skill", "DAS3H", "DASH", "IRT/MIRT", "PFA", "AFM"]


def compute_matrices(df, split_mode, metric="AUC", alpha=0.05):
    """Renvoie (delta_df, p_df, pcorr_df) : matrices carrées models×models.
    delta[a,b] = mean(a) - mean(b) apparié ;  p et p corrigé Holm."""
    sub = df[df.split_mode == split_mode]
    index_col = "global_fold" if "global_fold" in df.columns else "fold"
    pivot = sub.pivot_table(index=index_col, columns="model", values=metric)

    models = [m for m in MODEL_ORDER if m in pivot.columns]
    models += [m for m in pivot.columns if m not in models]  # au cas où

    delta = pd.DataFrame(np.nan, index=models, columns=models)
    pval  = pd.DataFrame(np.nan, index=models, columns=models)

    pairs, raw_p = [], []
    for a, b in combinations(models, 2):
        x, y = pivot[a].dropna(), pivot[b].dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]
        d = x.mean() - y.mean()
        delta.loc[a, b] = d
        delta.loc[b, a] = -d
        if len(x) < 1 or (x - y).abs().sum() == 0:
            continue
        _, p = wilcoxon(x, y)
        pairs.append((a, b))
        raw_p.append(p)

    # correction Holm sur les tests de CETTE famille (split × métrique)
    if raw_p:
        pcorr = multipletests(raw_p, alpha=alpha, method="holm")[1]
    else:
        pcorr = []

    pcorr_df = pd.DataFrame(np.nan, index=models, columns=models)
    for (a, b), p, pc in zip(pairs, raw_p, pcorr):
        pval.loc[a, b] = pval.loc[b, a] = p
        pcorr_df.loc[a, b] = pcorr_df.loc[b, a] = pc

    return delta, pval, pcorr_df

def plot_matrix(delta, pcorr, split_mode, metric, alpha=0.05, ax=None,
                vmax_abs=None):
    """Heatmap : couleur = Δ (taille d'effet, orienté 'ligne meilleure'),
    texte = Δ brut signé, * si p corrigé < alpha."""
    models = delta.index.tolist()
    n = len(models)

    # AUC : plus grand = mieux (+1). RMSE/NLL : plus petit = mieux (-1).
    # On oriente la COULEUR pour que rouge = 'ligne meilleure que colonne'
    # quelle que soit la métrique. Le TEXTE reste le Δ brut (non réorienté).
    orient = +1.0 if metric.upper() == "AUC" else -1.0
    delta_oriented = delta.values.astype(float) * orient

    # masque triangle supérieur + diagonale (façon matrice de corrélation)
    mask = np.triu(np.ones((n, n), dtype=bool))
    dm = np.ma.array(delta_oriented, mask=mask)

    # échelle symétrique autour de 0 ; plafond = max |Δ| observé (hors masque)
    if vmax_abs is None:
        finite = np.abs(delta_oriented[~mask & ~np.isnan(delta_oriented)])
        vmax_abs = finite.max() if finite.size else 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))

    cmap = plt.cm.RdBu_r.copy()   # rouge = positif (ligne meilleure), bleu = négatif
    cmap.set_bad("white")
    im = ax.imshow(dm, cmap=cmap, vmin=-vmax_abs, vmax=+vmax_abs, aspect="equal")

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            d = delta.values[i, j]          # Δ BRUT pour le texte (sens naturel)
            pc = pcorr.values[i, j]
            if np.isnan(d):
                continue
            star = "*" if (not np.isnan(pc) and pc < alpha) else ""
            # texte noir sur cases pâles, blanc sur cases saturées
            intensity = abs(delta_oriented[i, j]) / vmax_abs if vmax_abs else 0
            txt_color = "white" if intensity > 0.55 else "black"
            ax.text(j, i, f"{d:+.3f}{star}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title(f"{metric} — split {split_mode}\n"
                 f"(couleur = Δ orienté 'ligne meilleure', texte = Δ brut, * p<{alpha} Holm)",
                 fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Δ {metric} (rouge = ligne meilleure)", fontsize=8)
    return ax


if __name__ == "__main__":
    split_modes = ["user", "interaction"]
    metrics = ["AUC", "RMSE", "NLL"]
    ALPHA = 0.05

    out_dir = os.path.join(DATA_FOLDER, "figures")
    os.makedirs(out_dir, exist_ok=True)

    for split in split_modes:
        # une figure par split, 3 sous-graphes (une métrique chacun)
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        for ax, metric in zip(axes, metrics):
            print(f"\n=== wilcoxon | split:{split} | metric:{metric} ===")
            delta, pval, pcorr = compute_matrices(DF, split, metric, alpha=ALPHA)

            # affichage texte (comme avant, avec p corrigé)
            for a, b in combinations(delta.index, 2):
                d = delta.loc[a, b]; p = pval.loc[a, b]; pc = pcorr.loc[a, b]
                if np.isnan(p):
                    print(f"  {a} vs {b}: non testable"); continue
                sig = "SIGNIF" if pc < ALPHA else "ns"
                print(f"  {a} vs {b}: p={p:.4f} p_holm={pc:.4f} [{sig}]  Δ={d:+.4f}")

            plot_matrix(delta, pcorr, split, metric, alpha=ALPHA, ax=ax)

        fig.suptitle(f"Wilcoxon apparié multi-seed — {FOLDER.split('/')[0]} — split {split}",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        save_path = os.path.join(out_dir, f"wilcoxon_matrix_{split}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f">>> figure enregistrée : {save_path}")

    plt.show()
    print("done !!!")