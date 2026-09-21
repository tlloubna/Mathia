import os
import sys

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import src.Process.DAS3H as DAS3H
 # ("algebra05", 574)
    #("bridge_algebra06", 1146)
    #("Mathiadata", 25351),
    #("ASSISTments13_12", 15698),
     
N_STUDENTS = 25351
FOLDER = "Mathiadata"
DATA_FOLDER = os.path.join("data", FOLDER)
FIG_DIR = os.path.join(DATA_FOLDER+"/results", "figures")

MIN_OBS = 5       # nb minimal d'interactions (s,k) pour retenir la paire
N_KC = 6              # nb de compétences dans la grille (Figure A)
MAX_KC_B = 8          # nb max de compétences affichées en Figure B
ALPHA_GLOB_TOL = 0.05 # tolérance "alpha global quasi égal" (Figure B)

PATH_SKILL =  os.path.join(DATA_FOLDER, f"das3h_model_Alpha_C0.01_{N_STUDENTS}std.pkl")
PATH_GLOBAL = os.path.join(DATA_FOLDER, f"das3h_model_C0.01_{N_STUDENTS}std.pkl")
PATH_DF = os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv")
das3h_skill = joblib.load(PATH_SKILL)
model: DAS3H.DAS3HModel = das3h_skill["model"]
df = pd.read_csv(PATH_DF)
params = model.get_paramAlphask()
alpha_sk = params.get("alpha_sk")
def build_tab():
    emp = (df.groupby(["user_id", "KC"])["correct"].agg(success_rate="mean", n_obs="count").reset_index())
    emp["alpha"] = emp.apply(lambda r: alpha_sk.get((r["user_id"], r["KC"]), np.nan), axis=1)
    tab = emp.dropna(subset=["alpha"])
    tab = tab[(tab["n_obs"] >= MIN_OBS) & (tab["alpha"] != 0.0)].copy()
    return tab
def figure_A(tab):
    top_kc = tab["KC"].value_counts().head(N_KC).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, kc in zip(axes.flat, top_kc):
        sub = tab[tab["KC"] == kc]
        rho, p = spearmanr(sub["alpha"], sub["success_rate"])
        ax.scatter(sub["alpha"], sub["success_rate"], s=14, alpha=0.4,
                   color="steelblue", edgecolors="none")
        if len(sub) > 2:
            b, a = np.polyfit(sub["alpha"], sub["success_rate"], 1)
            xs = np.linspace(sub["alpha"].min(), sub["alpha"].max(), 50)
            ax.plot(xs, a + b * xs, color="crimson", lw=2)
        label = (kc[:40] + "…") if len(kc) > 40 else kc
        ax.set_title(f"{label}\nρ={rho:.2f}  (n={len(sub)})", fontsize=9)
        ax.set_xlabel(r"$\alpha_{s,k}$")
        ax.set_ylabel("taux réussite")
        ax.set_ylim(-0.05, 1.05)

    # cacher les axes restants si moins de 6 compétences
    for ax in axes.flat[len(top_kc):]:
        ax.set_visible(False)

    fig.suptitle("Figure A — Validation de α par compétence "
                 f"({len(top_kc)} compétences les plus pratiquées, n_obs≥{MIN_OBS})",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "figA_par_competence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f">>> {out}")
    plt.show()
def annotate(ax, y_pos, values, fmt, x_pad):
    """Écrit la valeur au bout de chaque barre horizontale."""
    for yi, val in zip(y_pos, values):
        x = val + x_pad if val >= 0 else val - x_pad
        ha = "left" if val >= 0 else "right"
        ax.text(x, yi, fmt.format(val), va="center", ha=ha, fontsize=7)


def figure_B(tab):
    # alpha global par élève (modèle DAS3H standard)
    g = joblib.load(PATH_GLOBAL)
    model_g: DAS3H.DAS3HModel = g["model"]
    params_g = model_g.get_params()          
    alpha_glob = params_g["alpha_s"]       

    # profils : matrice élève x KC des alpha_{s,k}, et des taux de réussite
    prof = tab.pivot_table(index="user_id", columns="KC", values="alpha")
    sr = tab.pivot_table(index="user_id", columns="KC", values="success_rate")
    users = [u for u in prof.index if u in alpha_glob]
    prof = prof.loc[users]
    P = prof.to_numpy()                      
    ag = np.array([alpha_glob[u] for u in users])
    order = np.argsort(ag)
    users_s = [users[i] for i in order]
    P_s = P[order]
    ag_s = ag[order]
    M = ~np.isnan(P_s)                        

    WINDOW = 200         
    MIN_COMMON = 4        
    best = None
    n = len(users_s)
    for i in range(n):
        j_max = min(i + WINDOW, n)
        # bornes alpha : dès que Δα dépasse la tolérance, inutile d'aller plus loin
        for j in range(i + 1, j_max):
            if ag_s[j] - ag_s[i] > ALPHA_GLOB_TOL:
                break                          # trié -> tous les suivants aussi
            common_mask = M[i] & M[j]
            n_common = common_mask.sum()
            if n_common < MIN_COMMON:
                continue
            diff = np.abs(P_s[i, common_mask] - P_s[j, common_mask]).mean()
            if best is None or diff > best[2]:
                best = (users_s[i], users_s[j], diff,
                        list(prof.columns[common_mask]))

    if best is None:
        print(f"[!] Aucun couple trouvé avec |Δα_global| ≤ {ALPHA_GLOB_TOL}. "
              f"Augmente ALPHA_GLOB_TOL ou WINDOW.")
        return

    u, v, diff, common = best
    print(f"Élèves {u} et {v} | α_global {alpha_glob[u]:.3f} vs {alpha_glob[v]:.3f} "
          f"| divergence profil moyenne={diff:.3f}")
    common = common[:MAX_KC_B]
    au = prof.loc[u, common].values          # alpha élève u
    av = prof.loc[v, common].values          # alpha élève v
    su = sr.loc[u, common].values            # taux réussite u
    sv = sr.loc[v, common].values            # taux réussite v

    y = np.arange(len(common)); h = 0.38
    labels = [(k[:32] + "…") if len(k) > 32 else k for k in common]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    # panneau gauche : alpha par compétence (maîtrise estimée)
    axL.barh(y - h / 2, au, height=h, color="steelblue",
             label=f"élève {u} (α_glob={alpha_glob[u]:.2f})")
    axL.barh(y + h / 2, av, height=h, color="darkorange",
             label=f"élève {v} (α_glob={alpha_glob[v]:.2f})")
    annotate(axL, y - h / 2, au, "{:+.2f}", 0.03)
    annotate(axL, y + h / 2, av, "{:+.2f}", 0.03)
    axL.axvline(0, color="black", lw=0.8)
    axL.set_xlabel(r"$\alpha_{s,k}$ appris (échelle logit)")
    axL.set_title("Maîtrise estimée par le modèle")
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=8)
    axL.margins(x=0.15)
    axL.legend(fontsize=8)

    # panneau droit : taux de réussite réel (performance observée)
    axR.barh(y - h / 2, su, height=h, color="steelblue")
    axR.barh(y + h / 2, sv, height=h, color="darkorange")
    annotate(axR, y - h / 2, su, "{:.0%}", 0.015)
    annotate(axR, y + h / 2, sv, "{:.0%}", 0.015)
    axR.set_xlim(0, 1.15)
    axR.set_xlabel("Taux de réussite empirique")
    axR.set_title("Performance réelle (données)")

    fig.suptitle("Figure B — Même aptitude globale, profils par compétence divergents\n"
                 "(la maîtrise estimée à gauche reflète la performance réelle à droite)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "figB_deux_eleves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f">>> {out}")
    plt.show()



def figure_C(tab):
    rho_s, p_s = spearmanr(tab["alpha"], tab["success_rate"])

    fig, ax = plt.subplots(figsize=(7.5, 6))

    hb = ax.scatter(tab["alpha"], tab["success_rate"],
                c=np.log10(tab["n_obs"]), cmap="viridis",
                s=18, alpha=0.6, edgecolors="none")
    # tendance linéaire globale
    b, a = np.polyfit(tab["alpha"], tab["success_rate"], 1)
    xs = np.linspace(tab["alpha"].min(), tab["alpha"].max(), 100)
    ax.plot(xs, a + b * xs, color="crimson", lw=2, label="tendance linéaire")

    ax.set_xlabel(r"$\alpha_{s,k}$ appris (échelle logit)")
    ax.set_ylabel("Taux de réussite empirique sur la compétence $k$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Figure C — Vue globale : α par compétence vs maîtrise réelle\n"
                 f"Spearman ρ={rho_s:.2f}  (n={len(tab)} paires, n_obs≥{MIN_OBS})",
                 fontsize=11)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("densité de paires (échelle log)")
    ax.legend()

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "figC_global_hexbin.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f">>> {out}")
    plt.show()
if __name__ == "__main__":
    figure = "C"          # "A" ou "B"
    tab = build_tab()
    print(f"Paires (s,k) retenues : {len(tab)} (seuil n_obs≥{MIN_OBS})")

    if figure == "A":
        figure_A(tab)
    elif figure == "B":
        figure_B(tab)

    elif figure=="C":
        figure_C(tab)

    print("done")