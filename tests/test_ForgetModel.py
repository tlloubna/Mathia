import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bornes alignées sur les fenêtres DAS3H : 1h, 1j, 1sem, 1mois, +
BINS = [0, 3600, 3600*24, 3600*24*7, 3600*24*30, np.inf]
BIN_LABELS = ["<1h", "1h-1j", "1j-7j", "7j-30j", ">30j"]


def _check_timestamp_unit(df):
    """Détecte si le timestamp est probablement en millisecondes et corrige.
    On regarde le delta médian entre interactions consécutives d'un même user.
    Un delta médian > ~1e5 en 'secondes' impliquerait >1 jour entre chaque clic,
    ce qui trahit des millisecondes."""
    d = df.sort_values(["user_id", "timestamp"])
    med = d.groupby("user_id")["timestamp"].diff().median()
    if med is not None and med > 1e5:
        print(f"[warn] delta médian = {med:.0f} : timestamp semble être en ms -> division par 1000")
        df = df.copy()
        df["timestamp"] = df["timestamp"] / 1000.0
    else:
        print(f"[ok] delta médian = {med:.0f} : timestamp traité comme des secondes")
    return df


def explode_kc(df):
    """Éclate les KC composites 'a~~b' en lignes distinctes, pour que delta_t
    soit calculé skill par skill (et non par combinaison de skills)."""
    df = df.copy()
    df["KC"] = df["KC"].astype(str).str.split("~~")
    df = df.explode("KC")
    df["KC"] = df["KC"].str.strip()
    return df


def compute_forgetting_curves(df, explode=True):
    """Calcule delta_t = temps depuis la dernière tentative du MÊME (user, KC).
    Vectorisé via groupby().diff() au lieu de iterrows()."""
    df = _check_timestamp_unit(df)
    if explode:
        df = explode_kc(df)
    df = df.sort_values(["user_id", "KC", "timestamp"])
    df["delta_t"] = df.groupby(["user_id", "KC"])["timestamp"].diff()
    # garde-fou : pas de delta négatif (tri/timestamps propres)
    df = df[(df["delta_t"].isna()) | (df["delta_t"] >= 0)]
    return df


def build_forgetting_curve(df):
    df = df.dropna(subset=["delta_t"]).copy()
    df["time_bin"] = pd.cut(df["delta_t"], bins=BINS, labels=BIN_LABELS,
                            include_lowest=True)
    grp = df.groupby("time_bin", observed=True)["correct"]
    curve = grp.mean()
    counts = grp.size()          # n par bin : indispensable pour juger la fiabilité
    return curve, counts


def plot(df,titre):
    curve, counts = build_forgetting_curve(df)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(curve.index.astype(str), curve.values, marker="o", linewidth=2)
    # annoter le n de chaque point
    for x, y, n in zip(curve.index.astype(str), curve.values, counts.values):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=11)
    ax.set_ylabel("Probabilité de réussite", fontsize=18)
    ax.set_xlabel("Temps depuis dernière révision", fontsize=18)
    ax.tick_params(axis="x", labelrotation=45, labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Forgetting curve of {titre} data ")
    
    fig.tight_layout()
    plt.show()
    return curve, counts


def plot_forgetting_per_student_curves_only(df, n_kc=10,titre=None):
    top_student = df["user_id"].value_counts().idxmax()
    print(f"Étudiant avec le plus d'interactions : {top_student}")
    df_student = df[df["user_id"] == top_student].dropna(subset=["delta_t"]).copy()
    df_student["time_bin"] = pd.cut(df_student["delta_t"], bins=BINS,
                                    labels=BIN_LABELS, include_lowest=True)
    curves = (df_student.groupby(["KC", "time_bin"], observed=True)["correct"]
              .mean().unstack(level=0))
    # ne garder que les KC presque complets sur les 5 bins
    curves_clean = curves.loc[:, curves.isna().sum() <= 1]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for kc in curves_clean.columns[:n_kc]:
        ax.plot(curves_clean.index.astype(str), curves_clean[kc],
                marker="o", linewidth=2)
    ax.set_ylabel("Probabilité de réussite", fontsize=18)
    ax.set_xlabel("Temps depuis dernière révision", fontsize=18)
    ax.tick_params(axis="x", labelrotation=45, labelsize=14)
    ax.set_title(f"Courbes d'oubli par KC — étudiant {top_student}", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(curves_clean.columns[:n_kc], title="KC", fontsize=10)
    ax.set_title(f"Forgetting curve of {titre} data ", fontsize=18)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    Liste_path=[("/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/preprocessed_data_simulated_1000std.csv","simulated"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/bridge_algebra06/preprocessed_data_1146std.csv","bridge"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata2/preprocessed_data_3178std.csv","mathdata"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/Mathiadata_v2/preprocessed_data_100000std.csv","mathia10000"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/mathia_dataCompleted_v2/preprocessed_data.csv","completedMathia"),
    ("/home/loubna/Code_Projet_Mathia/Mathia/data/ASSISTments13_12/preprocessed_data_15698std.csv","assist")]
    for path_titre in Liste_path: 
        df = pd.read_csv(path_titre[0])
        df = compute_forgetting_curves(df, explode=True)
        curve, counts = plot(df,path_titre[1])
        print(curve)
        print(counts)
        plot_forgetting_per_student_curves_only(df,titre=path_titre[1])
    print("Done!")