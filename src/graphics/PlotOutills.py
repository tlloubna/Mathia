import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from utils.this_queue import OurQueue

# Palette commune
BLUE   = "#4C9BE8"
PINK   = "#E85C8A"
GREEN  = "#3DBE8A"
ORANGE = "#F5A623"
PURPLE = "#9B59B6"
COLORS = [BLUE, PINK, GREEN, ORANGE, PURPLE, "#E74C3C", "#1ABC9C", "#F39C12"]
DARK_BG = "#1C1C2E"
CARD_BG = "#252540"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   "#3A3A5C",
    "axes.labelcolor":  "#CCCCDD",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#CCCCDD",
    "grid.color":       "#2E2E4E",
    "grid.linewidth":   0.6,
})


class PlotOUTILS:
    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────────────────
    # DÉJÀ EXISTANTS
    # ─────────────────────────────────────────────────────────────────────

    def plot_Q_matrix(self, Q=None, max_items=200):
        Q_small = Q[:max_items, :]
        plt.figure(figsize=(12, 8))
        sns.heatmap(Q_small, cmap="Greys", cbar=False)
        plt.xlabel("Skills (KC)")
        plt.ylabel("Items")
        plt.title(f"Matrix Q (display limited to {max_items} items)")
        plt.show()

    def PlotScoreDifficulty(self, score_diff=None):
        sns.heatmap(score_diff[['difficulty_score']], cmap="Reds", annot=False)
        plt.title("Heatmap des difficultés des KC")
        plt.show()

    def plot_all_forgetting_curves(self, forgettingDict):
        plt.figure(figsize=(10, 6))
        t = np.linspace(0, 40, 2)
        for kc, (a, b) in forgettingDict.items():
            P = 1 / (1 + np.exp(-(a - b * t)))
            plt.plot(t, P, alpha=0.3)
        plt.xlabel("Days")
        plt.ylabel("Probabilité de réussite")
        plt.title("Courbes d'oubli pour toutes les compétences")
        plt.grid()
        plt.show()

    def plotpresence(self, presence, vars):
        kcs    = list(presence.keys())
        counts = list(presence.values())
        plt.figure(figsize=(14, 6))
        plt.bar(kcs, counts, color="mediumseagreen", edgecolor="black")
        plt.xticks(rotation=90)
        plt.xlabel(f"{vars[0]}")
        plt.ylabel(f"{vars[1]}")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def PlotROC(self, TPR, FPR, AUC):
        plt.figure(figsize=(14, 6))
        plt.plot(FPR, TPR, alpha=0.6, color=BLUE, lw=2)
        plt.plot([0, 1], [0, 1], "--", color="#555566", lw=1)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC Curve  (AUC = {AUC:.3f})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    
    def PlotProbVsDifficulty(self, params: dict):
        """
        Trace P(correct) = sigmoid(alpha - delta + intercept)
        en faisant varier delta sur [-4, 4] pour plusieurs quantiles d'ability.
        Vérifie que la courbe décroît bien avec la difficulté.
        """
        intercept = params["intercept"]
        alphas_all = np.array(list(params["alpha_s"].values()))

        # 5 quantiles représentatifs
        quantiles  = [10, 25, 50, 75, 90]
        alpha_vals = np.percentile(alphas_all, quantiles)
        labels     = [f"α p{q} ({v:.2f})" for q, v in zip(quantiles, alpha_vals)]

        delta_range = np.linspace(-4, 4, 200)
        sigmoid     = lambda x: 1 / (1 + np.exp(-x))

        fig, ax = plt.subplots(figsize=(10, 5))
        for alpha, label, color in zip(alpha_vals, labels, COLORS):
            probs = sigmoid(alpha - delta_range + intercept)
            ax.plot(delta_range, probs, label=label, color=color, lw=2)

        ax.axhline(0.5, color="#555566", lw=1, linestyle="--", label="P = 0.5")
        ax.axvline(0,   color="#555566", lw=1, linestyle=":")
        ax.set_xlabel("Difficulté δⱼ")
        ax.set_ylabel("P(correct)")
        ax.set_title("P(correct) en fonction de la difficulté — par quantile d'ability")
        ax.legend(fontsize=9, framealpha=0.3)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Distributions ability (alpha_s) et difficulté (delta_j)
    # Usage : plot.PlotDistribParams(params)
    # ─────────────────────────────────────────────────────────────────────────
    def PlotDistribParams(self, params: dict):
        """
        Histogrammes + KDE des distributions alpha_s et delta_j.
        Permet de vérifier que les distributions sont cohérentes (ex : centrées sur 0).
        """
        alphas = np.array(list(params["alpha_s"].values()))
        deltas = np.array(list(params["delta_j"].values()))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, data, label, color in zip(
            axes,
            [alphas, deltas],
            ["Ability αₛ  (par élève)", "Difficulté δⱼ  (par item)"],
            [BLUE, PINK]
        ):
            ax.hist(data, bins=40, color=color, alpha=0.6, edgecolor="none", density=True)
            kde_x = np.linspace(data.min(), data.max(), 300)
            from scipy.stats import gaussian_kde
            kde   = gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color=color, lw=2.5)
            ax.axvline(np.mean(data),   color="white",  lw=1.5, linestyle="--", label=f"μ = {np.mean(data):.2f}")
            ax.axvline(np.median(data), color=ORANGE,   lw=1.5, linestyle=":",  label=f"med = {np.median(data):.2f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Densité")
            ax.legend(fontsize=9, framealpha=0.3)
            ax.grid(True, alpha=0.4)
            ax.set_title(label)

        fig.suptitle("Distributions des paramètres DAS3H estimés", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Courbes d'oubli — theta_wins et theta_attempts par KC × fenêtre temporelle
    # Usage : plot.PlotForgettingCurves(params, top_n=8)
    # ─────────────────────────────────────────────────────────────────────────
    def PlotForgettingCurves(self, params: dict, top_n: int = 8):
        """
        Pour chaque KC (top_n KC avec le plus grand theta_wins total),
        trace le coefficient theta en fonction de la fenêtre temporelle.
        Permet de voir quels KC ont un fort effet mémoire et comment il décroît.

        Fenêtres : [1h, 1j, 1sem, 1mois, ∞]
        """
        TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
        x = np.arange(len(TW_LABELS))

        theta_wins     = params["theta_wins"]
        theta_attempts = params["theta_attempts"]

        # Sélectionner les top_n KC selon la somme des theta_wins
        kc_scores = {kc: float(np.sum(np.abs(v))) for kc, v in theta_wins.items()}
        top_kcs   = sorted(kc_scores, key=kc_scores.get, reverse=True)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, theta_dict, title, marker in zip(
            axes,
            [theta_wins, theta_attempts],
            ["θ_wins : effet des succès passés", "θ_attempts : effet des tentatives passées"],
            ["o", "s"]
        ):
            for i, kc in enumerate(top_kcs):
                vals  = np.array(theta_dict[kc])
                label = kc[:30] + "…" if len(kc) > 30 else kc
                ax.plot(x, vals, marker=marker, color=COLORS[i % len(COLORS)],
                        lw=2, ms=6, label=label, alpha=0.85)

            ax.axhline(0, color="#555566", lw=1, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(TW_LABELS)
            ax.set_xlabel("Fenêtre temporelle")
            ax.set_ylabel("Coefficient θ")
            ax.set_title(title)
            ax.legend(fontsize=7, framealpha=0.3, loc="upper right")
            ax.grid(True, alpha=0.4)

        fig.suptitle("Courbes d'oubli DAS3H — Top KC par magnitude de θ", fontsize=13)
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Effet mémoire : log(1+n) × theta en fonction du nb de succès n
    # Usage : plot.PlotMemoryEffect(params, top_n=6)
    # ─────────────────────────────────────────────────────────────────────────
    def PlotMemoryEffect(self, params: dict, top_n: int = 6, tw_idx: int = 1):
        """
        Trace la contribution mémoire = log(1+n) × theta_wins[tw_idx]
        en fonction du nombre n de succès passés dans la fenêtre tw_idx.

        tw_idx : 0=1h, 1=1j (défaut), 2=1sem, 3=1mois, 4=∞
        """
        TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
        theta_wins = params["theta_wins"]

        kc_scores = {kc: float(np.abs(np.array(v)[tw_idx])) for kc, v in theta_wins.items()}
        top_kcs   = sorted(kc_scores, key=kc_scores.get, reverse=True)[:top_n]

        n_range = np.arange(0, 31)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, kc in enumerate(top_kcs):
            theta = float(np.array(theta_wins[kc])[tw_idx])
            mem   = np.log(1 + n_range) * theta
            label = (kc[:28] + "…" if len(kc) > 28 else kc) + f"  (θ={theta:.3f})"
            ax.plot(n_range, mem, color=COLORS[i % len(COLORS)], lw=2.2, label=label)

        ax.axhline(0, color="#555566", lw=1, linestyle="--")
        ax.set_xlabel(f"Nombre de succès n dans la fenêtre [{TW_LABELS[tw_idx]}]")
        ax.set_ylabel("Contribution mémoire  log(1+n) × θ")
        ax.set_title(f"Effet mémoire des succès passés — fenêtre {TW_LABELS[tw_idx]}")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. P(correct) d'un élève au fil de ses tentatives (simulation)
    # Usage : plot.PlotStudentLearning(params, user_id=0, item_id=5, kc_names=["KC1"])
    # ─────────────────────────────────────────────────────────────────────────
    def PlotStudentLearning(self, params: dict, user_id, item_id, kc_names: list,
                             n_attempts: int = 20, tw_idx: int = 1):
        """
        Simule l'évolution de P(correct) pour un élève sur un item donné
        au fur et à mesure des tentatives (toutes réussies → cas optimiste).

        Montre comment la mémoire (theta_wins) fait monter la proba.
        """
        TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
        sigmoid   = lambda x: 1 / (1 + np.exp(-x))

        alpha     = params["alpha_s"].get(user_id, 0.0)
        delta     = params["delta_j"].get(item_id, 0.0)
        intercept = params["intercept"]

        beta  = sum(params["beta_k"].get(kc, 0.0) for kc in kc_names)
        theta = sum(float(np.array(params["theta_wins"][kc])[tw_idx])
                    for kc in kc_names if kc in params["theta_wins"])

        attempts = np.arange(0, n_attempts + 1)
        memory   = np.log(1 + attempts) * theta
        logits   = alpha - delta + beta + memory + intercept
        probs    = sigmoid(logits)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Courbe P(correct) vs tentatives
        axes[0].plot(attempts, probs, color=BLUE, lw=2.5, marker="o", ms=5)
        axes[0].axhline(0.5, color="#555566", lw=1, linestyle="--", label="seuil 0.5")
        axes[0].fill_between(attempts, 0.5, probs,
                             where=probs >= 0.5, alpha=0.15, color=GREEN, label="zone maîtrisée")
        axes[0].set_xlabel("Nombre de tentatives réussies")
        axes[0].set_ylabel("P(correct)")
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f"Évolution de P(correct)\nÉlève {user_id} · Item {item_id}")
        axes[0].legend(fontsize=9, framealpha=0.3)
        axes[0].grid(True, alpha=0.4)

        # Décomposition des contributions
        contrib = {
            "Ability α":    alpha + intercept,
            "Difficulté −δ": -delta,
            "Easiness β_k": beta,
            f"Mémoire (n=5)": np.log(1 + 5) * theta,
        }
        bars   = list(contrib.values())
        labels = list(contrib.keys())
        bar_colors = [BLUE if v >= 0 else PINK for v in bars]

        axes[1].barh(labels, bars, color=bar_colors, alpha=0.8, edgecolor="none")
        axes[1].axvline(0, color="white", lw=1)
        axes[1].set_xlabel("Contribution au logit")
        axes[1].set_title("Décomposition du logit (n=5 succès)")
        axes[1].grid(True, alpha=0.4, axis="x")
        for i, (v, label) in enumerate(zip(bars, labels)):
            axes[1].text(v + 0.03 * np.sign(v), i, f"{v:.3f}",
                         va="center", fontsize=9, color="white")

        fig.suptitle(f"Simulation apprentissage DAS3H — KC : {', '.join(kc_names[:3])}", fontsize=12)
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Heatmap theta_wins : KC × fenêtres temporelles
    # Usage : plot.PlotThetaHeatmap(params, top_n=20)
    # ─────────────────────────────────────────────────────────────────────────
    def PlotThetaHeatmap(self, params: dict, top_n: int = 20):
        """
        Heatmap des coefficients theta_wins pour les top_n KC.
        Permet de repérer d'un coup d'œil quels KC ont un fort effet mémoire
        et sur quelle fenêtre temporelle.
        """
        TW_LABELS  = ["1h", "1j", "1sem", "1mois", "∞"]
        theta_wins = params["theta_wins"]

        kc_scores = {kc: float(np.sum(np.abs(v))) for kc, v in theta_wins.items()}
        top_kcs   = sorted(kc_scores, key=kc_scores.get, reverse=True)[:top_n]

        matrix = np.array([list(theta_wins[kc]) for kc in top_kcs])
        labels = [kc[:35] + "…" if len(kc) > 35 else kc for kc in top_kcs]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                       vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())

        ax.set_xticks(range(len(TW_LABELS)))
        ax.set_xticklabels(TW_LABELS, fontsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Fenêtre temporelle")
        ax.set_title(f"Heatmap θ_wins — Top {top_n} KC par magnitude")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.3 * np.abs(matrix).max() else "#888899")

        plt.colorbar(im, ax=ax, shrink=0.6, label="θ coefficient")
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Dashboard complet — tout en un seul appel
    # Usage : plot.PlotDAS3HDashboard(params)
    # ─────────────────────────────────────────────────────────────────────────
    def PlotDAS3HDashboard(self, params: dict, top_n_kc: int = 6):
        """
        Résumé visuel complet des paramètres DAS3H en une seule figure :
        • Distribution ability / difficulté
        • P(correct) vs difficulté
        • Courbes d'oubli theta_wins
        • Heatmap theta_wins × KC
        """
        from scipy.stats import gaussian_kde
        TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]

        alphas = np.array(list(params["alpha_s"].values()))
        deltas = np.array(list(params["delta_j"].values()))
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        kc_scores = {kc: float(np.sum(np.abs(v))) for kc, v in params["theta_wins"].items()}
        top_kcs   = sorted(kc_scores, key=kc_scores.get, reverse=True)[:top_n_kc]

        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── A : Distribution ability ──────────────────────────────────────
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.hist(alphas, bins=30, color=BLUE, alpha=0.6, edgecolor="none", density=True)
        kde_x = np.linspace(alphas.min(), alphas.max(), 300)
        ax_a.plot(kde_x, gaussian_kde(alphas)(kde_x), color=BLUE, lw=2)
        ax_a.axvline(np.mean(alphas), color="white", lw=1.5, linestyle="--",
                     label=f"μ={np.mean(alphas):.2f}")
        ax_a.set_title("Distribution Ability αₛ")
        ax_a.set_xlabel("αₛ")
        ax_a.legend(fontsize=8, framealpha=0.3)
        ax_a.grid(True, alpha=0.3)

        # ── B : Distribution difficulté ───────────────────────────────────
        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.hist(deltas, bins=40, color=PINK, alpha=0.6, edgecolor="none", density=True)
        kde_x2 = np.linspace(deltas.min(), deltas.max(), 300)
        ax_b.plot(kde_x2, gaussian_kde(deltas)(kde_x2), color=PINK, lw=2)
        ax_b.axvline(np.mean(deltas), color="white", lw=1.5, linestyle="--",
                     label=f"μ={np.mean(deltas):.2f}")
        ax_b.set_title("Distribution Difficulté δⱼ")
        ax_b.set_xlabel("δⱼ")
        ax_b.legend(fontsize=8, framealpha=0.3)
        ax_b.grid(True, alpha=0.3)

        # ── C : P(correct) vs difficulté ─────────────────────────────────
        ax_c = fig.add_subplot(gs[0, 2])
        delta_range  = np.linspace(-4, 4, 200)
        alpha_quants = np.percentile(alphas, [10, 50, 90])
        q_labels     = ["α p10", "α médian", "α p90"]
        for alpha_q, ql, col in zip(alpha_quants, q_labels, [PINK, BLUE, GREEN]):
            ax_c.plot(delta_range, sigmoid(alpha_q - delta_range + params["intercept"]),
                      label=f"{ql} ({alpha_q:.2f})", color=col, lw=2)
        ax_c.axhline(0.5, color="#555566", lw=1, linestyle="--")
        ax_c.set_xlabel("Difficulté δⱼ")
        ax_c.set_ylabel("P(correct)")
        ax_c.set_ylim(0, 1)
        ax_c.set_title("P(correct) vs Difficulté")
        ax_c.legend(fontsize=8, framealpha=0.3)
        ax_c.grid(True, alpha=0.3)

        # ── D : Courbes d'oubli θ_wins ────────────────────────────────────
        ax_d = fig.add_subplot(gs[1, 0:2])
        x = np.arange(len(TW_LABELS))
        for i, kc in enumerate(top_kcs):
            vals  = np.array(params["theta_wins"][kc])
            label = kc[:32] + "…" if len(kc) > 32 else kc
            ax_d.plot(x, vals, marker="o", color=COLORS[i % len(COLORS)],
                      lw=2, ms=6, label=label)
        ax_d.axhline(0, color="#555566", lw=1, linestyle="--")
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(TW_LABELS)
        ax_d.set_xlabel("Fenêtre temporelle")
        ax_d.set_ylabel("θ_wins")
        ax_d.set_title(f"Courbes d'oubli θ_wins — Top {top_n_kc} KC")
        ax_d.legend(fontsize=7, framealpha=0.3, loc="upper right")
        ax_d.grid(True, alpha=0.3)

        # ── E : Heatmap θ_wins ────────────────────────────────────────────
        ax_e = fig.add_subplot(gs[1, 2])
        matrix = np.array([list(params["theta_wins"][kc]) for kc in top_kcs])
        short_labels = [kc[:20] + "…" if len(kc) > 20 else kc for kc in top_kcs]
        vmax = np.abs(matrix).max()
        im   = ax_e.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax_e.set_xticks(range(len(TW_LABELS)))
        ax_e.set_xticklabels(TW_LABELS, fontsize=9)
        ax_e.set_yticks(range(len(short_labels)))
        ax_e.set_yticklabels(short_labels, fontsize=7)
        ax_e.set_title("Heatmap θ_wins")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax_e.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if abs(matrix[i,j]) > 0.4*vmax else "#aaaaaa")
        plt.colorbar(im, ax=ax_e, shrink=0.7)

        fig.suptitle("DAS3H — Tableau de bord des paramètres estimés", fontsize=15, y=1.01)
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # PlotStudentTrajectory
    # Simule et visualise l'apprentissage + oubli d'un étudiant réel
    # sur ses 5 premiers KC, en rejouant ses vraies interactions depuis df.
    #
    # Usage :
    #   plot.PlotStudentTrajectory(params, df, user_id=3, top_n_kc=5)
    # ─────────────────────────────────────────────────────────────────────────
    def PlotStudentTrajectory(self, params: dict, df, user_id, top_n_kc: int = 5):
        """
        Pour un étudiant user_id, rejoue chronologiquement ses interactions
        et calcule à chaque instant :
          - P(correct) réelle  = sigmoid(alpha - delta + beta_k + mémoire(t))
          - La contribution mémoire wins  cumulée par KC
          - La contribution mémoire fails cumulée par KC

        Paramètres
        ----------
        params   : dict retourné par model.get_params()
        df       : DataFrame avec colonnes [user_id, item_id, KC, timestamp, correct]
        user_id  : identifiant de l'étudiant à visualiser
        top_n_kc : nombre de KC à afficher (les plus fréquentes chez cet élève)
        """
        import pandas as pd
        from collections import defaultdict

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        TW_SECONDS = [3600, 86400, 604800, 2592000, float("inf")]
        TW_LABELS  = ["1h",  "1j",  "1sem", "1mois", "∞"]

        # ── 1. Filtrer les interactions de l'étudiant ──────────────────────
        df_stud = (
            df[df["user_id"] == user_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if df_stud.empty:
            print(f"[ERREUR] Aucune interaction trouvée pour user_id={user_id}")
            return

        # ── 2. Sélectionner les top_n_kc KC les plus fréquentes ───────────
        kc_counts  = df_stud["KC"].str.split("~~").explode().value_counts()
        top_kcs    = kc_counts.index[:top_n_kc].tolist()
        # Ne garder que les KC présentes dans params
        top_kcs    = [kc for kc in top_kcs if kc in params.get("beta_k", {})][:top_n_kc]

        if not top_kcs:
            print("[ERREUR] Aucun KC de cet élève n'est dans params. Vérifiez kc_list.")
            return

        alpha     = params["alpha_s"].get(user_id, 0.0)
        intercept = params["intercept"]

        # ── 3. Rejouer les interactions chronologiquement ──────────────────
        # Pour chaque KC on stocke les timestamps de succès et d'échecs
        history_wins  = defaultdict(list)   # timestamps des succès
        history_fails = defaultdict(list)   # timestamps des échecs

        # Résultats par KC
        records = {kc: [] for kc in top_kcs}

        for _, row in df_stud.iterrows():
            t       = float(row["timestamp"])
            correct = int(row["correct"])
            item_id = row["item_id"]
            kcs_row = str(row["KC"]).split("~~")

            for kc in top_kcs:
                if kc not in kcs_row:
                    continue  # cet item ne concerne pas ce KC

                # -- Récupérer les paramètres --
                delta   = params["delta_j"].get(item_id, 0.0)
                beta    = params["beta_k"].get(kc, 0.0)
                tw_wins = np.array(params["theta_wins"].get(kc,     np.zeros(5)))
                tw_att  = np.array(params["theta_attempts"].get(kc, np.zeros(5)))
                tf      = float(params["theta_fails"].get(kc, 0.0))

                # -- Compter les événements dans chaque fenêtre temporelle --
                def count_in_windows(timestamps, t_now):
                    counts = []
                    for tw in TW_SECONDS:
                        if tw == float("inf"):
                            counts.append(len(timestamps))
                        else:
                            counts.append(sum(1 for ts in timestamps if (t_now - ts) <= tw))
                    return np.array(counts, dtype=float)

                n_wins_tw  = count_in_windows(history_wins[kc],  t)
                n_fails    = len(history_fails[kc])

                # -- Contributions mémoire --
                mem_wins  = np.dot(tw_wins, np.log(1 + n_wins_tw))
                mem_att   = np.dot(tw_att,  np.log(1 + n_wins_tw))  # approx : attempts ≈ wins
                mem_fails = tf * n_fails

                # -- Logit et proba --
                logit = alpha - delta + beta + mem_wins + mem_att + mem_fails + intercept
                prob  = sigmoid(logit)

                records[kc].append({
                    "t":         t,
                    "prob":      prob,
                    "correct":   correct,
                    "mem_wins":  mem_wins,
                    "mem_fails": mem_fails,
                    "n_wins":    int(n_wins_tw[-1]),   # total wins (fenêtre ∞)
                    "n_fails":   n_fails,
                    "logit":     logit,
                })

                # -- Mettre à jour l'historique APRÈS calcul --
                if correct:
                    history_wins[kc].append(t)
                else:
                    history_fails[kc].append(t)

        # ── 4. Filtrer les KC avec au moins 2 interactions ─────────────────
        top_kcs = [kc for kc in top_kcs if len(records[kc]) >= 2]

        if not top_kcs:
            print("[ERREUR] Pas assez d'interactions pour tracer.")
            return

        # ── 5. Figure ──────────────────────────────────────────────────────
        n_kc = len(top_kcs)
        fig, axes = plt.subplots(n_kc, 3, figsize=(18, 3.5 * n_kc),
                                 sharex=False)
        if n_kc == 1:
            axes = [axes]

        # Convertir timestamps en heures relatives
        t0 = df_stud["timestamp"].min()

        for row_idx, kc in enumerate(top_kcs):
            data = records[kc]
            ts       = np.array([(d["t"] - t0) / 3600 for d in data])   # en heures
            probs    = np.array([d["prob"]      for d in data])
            corrects = np.array([d["correct"]   for d in data])
            mem_w    = np.array([d["mem_wins"]  for d in data])
            mem_f    = np.array([d["mem_fails"] for d in data])
            n_wins   = np.array([d["n_wins"]    for d in data])
            n_fails  = np.array([d["n_fails"]   for d in data])

            short_kc = kc[:40] + "…" if len(kc) > 40 else kc
            color    = COLORS[row_idx % len(COLORS)]

            # ── Colonne A : P(correct) au fil du temps ─────────────────────
            ax_p = axes[row_idx][0]
            ax_p.plot(ts, probs, color=color, lw=2, zorder=3)
            ax_p.fill_between(ts, 0, probs, alpha=0.12, color=color)
            ax_p.axhline(0.5, color="#555566", lw=1, linestyle="--", alpha=0.7)

            # Points colorés selon succès/échec
            for i, (t_i, p_i, c_i) in enumerate(zip(ts, probs, corrects)):
                marker_color = GREEN if c_i == 1 else PINK
                ax_p.scatter(t_i, p_i, color=marker_color, s=50, zorder=5,
                             edgecolors="white", linewidths=0.5)

            ax_p.set_ylim(0, 1)
            ax_p.set_ylabel("P(correct)", fontsize=9)
            ax_p.set_title(f"KC : {short_kc}\nP(correct) au fil du temps", fontsize=9)
            ax_p.grid(True, alpha=0.3)
            ax_p.set_xlabel("Temps (heures depuis début)", fontsize=8)

            # Légende succès/échec
            ax_p.scatter([], [], color=GREEN, s=40, label="✓ succès", edgecolors="white", lw=0.5)
            ax_p.scatter([], [], color=PINK,  s=40, label="✗ échec",  edgecolors="white", lw=0.5)
            ax_p.legend(fontsize=7, framealpha=0.3, loc="lower right")

            # ── Colonne B : Contributions mémoire ──────────────────────────
            ax_m = axes[row_idx][1]
            ax_m.plot(ts, mem_w, color=GREEN,  lw=2, label="mémoire wins",  marker="o", ms=4)
            ax_m.plot(ts, mem_f, color=PINK,   lw=2, label="mémoire fails", marker="s", ms=4,
                      linestyle="--")
            ax_m.fill_between(ts, 0, mem_w, alpha=0.1, color=GREEN)
            ax_m.fill_between(ts, mem_f, 0, alpha=0.1, color=PINK)
            ax_m.axhline(0, color="#555566", lw=1)
            ax_m.set_title("Contributions mémoire", fontsize=9)
            ax_m.set_ylabel("θ × log(1+n)", fontsize=9)
            ax_m.set_xlabel("Temps (heures)", fontsize=8)
            ax_m.legend(fontsize=7, framealpha=0.3)
            ax_m.grid(True, alpha=0.3)

            # ── Colonne C : Compteurs cumulés ──────────────────────────────
            ax_c = axes[row_idx][2]
            ax_c.step(ts, n_wins,  color=GREEN, lw=2, label="nb succès cumulés",  where="post")
            ax_c.step(ts, n_fails, color=PINK,  lw=2, label="nb échecs cumulés",  where="post",
                      linestyle="--")
            ax_c.set_title("Succès & échecs cumulés", fontsize=9)
            ax_c.set_ylabel("Nombre d'événements", fontsize=9)
            ax_c.set_xlabel("Temps (heures)", fontsize=8)
            ax_c.legend(fontsize=7, framealpha=0.3)
            ax_c.grid(True, alpha=0.3)

        # ── En-tête global ──────────────────────────────────────────────────
        fig.suptitle(
            f"Trajectoire d'apprentissage — Élève {user_id}  "
            f"(αₛ = {alpha:.3f})  ·  {len(df_stud)} interactions totales",
            fontsize=13, y=1.01
        )
        plt.tight_layout()
        plt.show()

        # ── Résumé texte ────────────────────────────────────────────────────
        print(f"\n{'═'*55}")
        print(f"  Résumé élève {user_id}  |  ability αₛ = {alpha:.3f}")
        print(f"{'─'*55}")
        for kc in top_kcs:
            data   = records[kc]
            p_init = data[0]["prob"]
            p_last = data[-1]["prob"]
            n_w    = data[-1]["n_wins"]
            n_f    = data[-1]["n_fails"]
            mastered = "✓ maîtrisé" if p_last >= 0.5 else "✗ non maîtrisé"
            print(f"  {kc[:35]:<35}  {mastered}")
            print(f"    P init={p_init:.3f} → P final={p_last:.3f}  "
                  f"| wins={n_w}  fails={n_f}")
        print(f"{'═'*55}\n")