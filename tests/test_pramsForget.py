from pathlib import Path
import os 
import sys 
# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
import matplotlib.pyplot as plt

# Ordre des fenêtres tel que produit par HistoryDATA / utilisé dans plot_real_theta_vs_future
WINDOW_LABELS =["1h", "1j", "1sem", "1mois", "∞"]


def stack_theta(params, key):
    """Empile les vecteurs theta de tous les KC en une matrice (n_kc, n_tw).
    key = 'theta_wins' ou 'theta_attempts' (les deux ont une structure par fenêtre)."""
    kcs = list(params[key].keys())
    mat = np.vstack([np.asarray(params[key][kc], dtype=float) for kc in kcs])
    return kcs, mat  # mat[i] = vecteur 5 fenêtres du skill i


def plot_forgetting_profile(params, labels=WINDOW_LABELS,titre=None):
    """Profil moyen des poids par fenêtre temporelle.
    Décroissance de 1h vers ∞  =>  DAS3H a capté l'oubli."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    x = np.arange(len(labels))

    for key, color in [("theta_wins", "steelblue"),
                       ("theta_attempts", "salmon")]:
        _, mat = stack_theta(params, key)
        mean_w = mat.mean(axis=0)
        std_w = mat.std(axis=0)
        ax.errorbar(x, mean_w, yerr=std_w, marker="o", capsize=4,
                    linewidth=2, color=color, label=key)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlabel("Fenêtre temporelle (du plus récent au plus ancien)", fontsize=16)
    ax.set_ylabel("Poids θ moyen sur les skills", fontsize=16)
    ax.set_title(f"Profil des poids DAS3H par fenêtre pour data :{titre}\n"
                 "décroissance 1h → ∞ = oubli capté", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_forgetting_heatmap(params, key="theta_wins", labels=WINDOW_LABELS, n_kc=30,titre=None):
    """Heatmap skill x fenêtre : voir si l'oubli est homogène entre skills
    ou concentré sur certains. n_kc limite l'affichage aux premiers skills."""
    kcs, mat = stack_theta(params, key)
    mat = mat[:n_kc]
    fig, ax = plt.subplots(figsize=(8, max(6, n_kc * 0.25)), dpi=120)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(len(kcs[:n_kc])))
    ax.set_yticklabels(kcs[:n_kc], fontsize=7)
    ax.set_xlabel("Fenêtre temporelle", fontsize=14)
    ax.set_ylabel("KC", fontsize=14)
    ax.set_title(f"{key} par skill et fenêtre pour data {titre}", fontsize=14)
    fig.colorbar(im, ax=ax, label="poids θ")
    fig.tight_layout()
    plt.show()


def forgetting_summary(params, labels=WINDOW_LABELS,titre=None):
    """Chiffre la décroissance : pente moyenne et ratio 1h / ∞."""
    print(f"explore data  {titre} ")
    for key in ["theta_wins", "theta_attempts"]:
        _, mat = stack_theta(params, key)
        mean_w = mat.mean(axis=0)
        slope = np.polyfit(np.arange(len(labels)), mean_w, 1)[0]
        print(f"\n{key}")
        for lab, v in zip(labels, mean_w):
            print(f"  {lab:>5} : {v:+.4f}")
        print(f"  pente (régression sur l'index de fenêtre) : {slope:+.4f}")
        print(f"  -> {'décroissant (oubli)' if slope < 0 else 'croissant/plat (pas d oubli net)'}")
if __name__ == "__main__":
    import joblib
    Liste_path=[("/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/das3h_model_C1_das3h_original.pkl","simulated"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/bridge_algebra06/das3h_model_C1_1146std.pkl","bridge"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata2/das3h_model_C1_3178std.pkl","mathdata"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/mathia_dataCompleted_v2/das3h_model_C1_das3h_original.pkl","completedMathia"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/ASSISTments13_12/das3h_model_C1_15698std.pkl","assist")]
    for path_titre in Liste_path: 
        loaded = joblib.load(path_titre[0])
        model = loaded["model"]
        params = model.get_params()
        forgetting_summary(params,titre=path_titre[1])
        plot_forgetting_profile(params,titre=path_titre[1])
        plot_forgetting_heatmap(params, key='theta_wins',titre=path_titre[1])
    print("done")

