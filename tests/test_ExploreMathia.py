
import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import sparse
from ast import literal_eval


DATA_FOLDER  = os.path.join("data", "Mathiadata")
N_STUDENTS   = 25351
CSV_PATH     = os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv")

###############Q matix #########################""
"""Q = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
print(f"Q-matrice : {Q.shape[0]} items × {Q.shape[1]} KCs")
print(f"Densité   : {Q.mean()*100:.2f}%")

plt.figure(figsize=(12, 8))
plt.imshow(Q, aspect="auto", cmap="Blues", interpolation="none")
plt.title(f"Q-matrice — {Q.shape[0]} items × {Q.shape[1]} KCs")
plt.xlabel("KCs")
plt.ylabel("Items (exercices)")

plt.tight_layout()
plt.show()"""
df = pd.read_csv(CSV_PATH)


print("Chargement des données...")


# timestamp en datetime
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
df["week"]     = df["datetime"].dt.to_period("W").dt.start_time
df["month"]    = df["datetime"].dt.to_period("M").dt.start_time


# Parser kc_names : "Addition" ou "Addition~~Soustraction"
df["kc_names_list"] = df["kc_names"].apply(
    lambda x: str(x).split("~~") if pd.notna(x) else []
)
df["kc_main"] = df["kc_names_list"].apply(lambda x: x[0] if len(x) > 0 else None)


df_kc = df.explode("kc_names_list").rename(columns={"kc_names_list": "kc"})
df_kc = df_kc[df_kc["kc"].notna() & (df_kc["kc"] != "")]

print(f"KCs uniques après explosion : {df_kc['kc'].nunique()}")

print(f"  {len(df):,} interactions | {df['user_id'].nunique():,} élèves | {df_kc['kc'].nunique()} KCs")
print(f"  Période : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"  Taux de réussite global : {df['correct'].mean()*100:.1f}%\n")



print("── AXE 1 : Vue globale ──")

"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Vue globale — Distributions", fontsize=14, fontweight="bold")"""

# 1a. Distribution interactions par élève
inter_per_student = df.groupby("user_id").size()
"""axes[0].hist(inter_per_student, bins=50, color="steelblue", edgecolor="white", log=True)
axes[0].axvline(inter_per_student.median(), color="red", linestyle="--",
                label=f"médiane={inter_per_student.median():.0f}")
axes[0].set_title("Interactions par élève (log)")
axes[0].set_xlabel("Nombre d'interactions")
axes[0].set_ylabel("Nombre d'élèves (log)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)"""

# 1b. Distribution interactions par KC
inter_per_kc = df_kc.groupby("kc").size().sort_values(ascending=False)
""""axes[1].bar(range(len(inter_per_kc)), inter_per_kc.values, color="salmon", width=1.0)
axes[1].set_title(f"Interactions par KC ({len(inter_per_kc)} KCs)")
axes[1].set_xlabel("KCs (triés par fréquence)")
axes[1].set_ylabel("Nombre d'interactions")
axes[1].axhline(1000, color="orange", linestyle="--", label="seuil 1000")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# 1c. Taux de réussite par KC
success_per_kc = df_kc.groupby("kc")["correct"].mean().sort_values()
axes[2].barh(range(len(success_per_kc)), success_per_kc.values,
             color=["#e74c3c" if v < 0.6 else "#2ecc71" for v in success_per_kc.values])
axes[2].axvline(success_per_kc.mean(), color="navy", linestyle="--",
                label=f"moyenne={success_per_kc.mean():.2f}")
axes[2].set_title("Taux de réussite par KC")
axes[2].set_xlabel("Taux de réussite")
axes[2].set_ylabel("KCs (triés)")
axes[2].set_yticks([])
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()

plt.show()"""



print("── AXE 2 : Évolution temporelle ──")
"""
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Évolution temporelle", fontsize=14, fontweight="bold")"""


weekly = df.groupby("week").agg(
    n_interactions=("correct", "count"),
    taux_reussite =("correct", "mean")
).reset_index()

"""ax1 = axes[0]
ax2 = ax1.twinx()
ax1.bar(weekly["week"], weekly["n_interactions"],
        width=5, color="steelblue", alpha=0.6, label="Interactions")
ax2.plot(weekly["week"], weekly["taux_reussite"],
         color="darkorange", linewidth=2, marker="o", markersize=4, label="Taux réussite")
ax1.set_title("Activité hebdomadaire")
ax1.set_xlabel("Semaine")
ax1.set_ylabel("Nombre d'interactions", color="steelblue")
ax2.set_ylabel("Taux de réussite", color="darkorange")
ax2.set_ylim(0, 1)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.grid(True, alpha=0.3)"""

# 2b. Heatmap élèves actifs par mois (top 50 élèves les plus actifs)
top_students = inter_per_student.nlargest(50).index
df_top = df[df["user_id"].isin(top_students)].copy()
pivot = df_top.pivot_table(
    index="user_id", columns="month",
    values="correct", aggfunc="count", fill_value=0
)
"""sns.heatmap(pivot, ax=axes[1], cmap="YlOrRd",
            xticklabels=[str(c.date())[:7] for c in pivot.columns],
            yticklabels=False, cbar_kws={"label": "Interactions"})
axes[1].set_title("Heatmap activité — Top 50 élèves (interactions/mois)")
axes[1].set_xlabel("Mois")
axes[1].set_ylabel("Élèves")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()

plt.show()
"""


print("── AXE 3 : Profils élèves ──")

"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Profils élèves", fontsize=14, fontweight="bold")"""

student_stats = df.groupby("user_id").agg(
    n_interactions =("correct", "count"),
    taux_reussite  =("correct", "mean"),
    n_kcs          =("kc_main", "nunique"),
    span_days      =("timestamp", lambda x: (x.max() - x.min()) / 86400)
).reset_index()

# 3a. Scatter interactions vs taux de réussite
"""sc = axes[0].scatter(
    student_stats["n_interactions"],
    student_stats["taux_reussite"],
    c=student_stats["n_kcs"], cmap="viridis",
    alpha=0.4, s=15
)
plt.colorbar(sc, ax=axes[0], label="Nb KCs travaillés")
axes[0].axhline(0.6, color="red", linestyle="--", alpha=0.5, label="seuil 60%")
axes[0].set_xlabel("Nombre d'interactions")
axes[0].set_ylabel("Taux de réussite")
axes[0].set_title("Interactions vs Réussite\n(couleur = nb KCs)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 3b. Distribution du taux de réussite par élève
axes[1].hist(student_stats["taux_reussite"], bins=40,
             color="mediumseagreen", edgecolor="white")
axes[1].axvline(student_stats["taux_reussite"].mean(), color="red",
                linestyle="--", label=f"moyenne={student_stats['taux_reussite'].mean():.2f}")
axes[1].axvline(0.6, color="orange", linestyle=":", label="seuil 60%")
axes[1].set_title("Distribution taux de réussite par élève")
axes[1].set_xlabel("Taux de réussite")
axes[1].set_ylabel("Nombre d'élèves")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3c. Distribution du nombre de KCs travaillés par élève
axes[2].hist(student_stats["n_kcs"], bins=30,
             color="mediumpurple", edgecolor="white")
axes[2].axvline(student_stats["n_kcs"].median(), color="red",
                linestyle="--", label=f"médiane={student_stats['n_kcs'].median():.0f}")
axes[2].set_title("Nombre de KCs distincts par élève")
axes[2].set_xlabel("Nombre de KCs")
axes[2].set_ylabel("Nombre d'élèves")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

plt.show()
"""



print("── AXE 4 : Courbes d'apprentissage ──")

# 4a. Courbe d'apprentissage globale : taux réussite à la Nème tentative sur un KC
print("  Calcul des numéros de tentative par (élève, KC)...")
df_sorted = df.sort_values(["user_id", "kc_main", "timestamp"])
df_sorted["attempt_n"] = df_sorted.groupby(["user_id", "kc_main"]).cumcount() + 1

# Moyenne sur tous élèves/KCs, jusqu'à la 20ème tentative
learning_curve = df_sorted[df_sorted["attempt_n"] <= 20].groupby("attempt_n").agg(
    mean_correct=("correct", "mean"),
    std_correct =("correct", "std"),
    n           =("correct", "count")
).reset_index()
learning_curve["se"] = learning_curve["std_correct"] / np.sqrt(learning_curve["n"])

"""fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Courbes d'apprentissage", fontsize=14, fontweight="bold")

axes[0].plot(learning_curve["attempt_n"], learning_curve["mean_correct"],
             marker="o", color="steelblue", linewidth=2)
axes[0].fill_between(
    learning_curve["attempt_n"],
    (learning_curve["mean_correct"] - learning_curve["se"]).clip(0, 1),
    (learning_curve["mean_correct"] + learning_curve["se"]).clip(0, 1),
    alpha=0.2, color="steelblue"
)
axes[0].axhline(df["correct"].mean(), color="red", linestyle="--",
                label=f"taux global={df['correct'].mean():.2f}")
axes[0].set_title("Courbe d'apprentissage globale\n(taux réussite à la Nème tentative sur un KC)")
axes[0].set_xlabel("Numéro de tentative sur le KC")
axes[0].set_ylabel("Taux de réussite moyen")
axes[0].set_ylim(0.5, 1.0)
axes[0].legend()
axes[0].grid(True, alpha=0.3)"""

# 4b. Courbes pour les 6 KCs les plus fréquents
top6_kcs = inter_per_kc.head(6).index.tolist()
colors = plt.cm.tab10(np.linspace(0, 1, 6))

"""for i, kc in enumerate(top6_kcs):
    lc_kc = df_sorted[
        (df_sorted["kc_main"] == kc) & (df_sorted["attempt_n"] <= 15)
    ].groupby("attempt_n")["correct"].mean()
    axes[1].plot(lc_kc.index, lc_kc.values,
                 marker="o", markersize=4, linewidth=2,
                 color=colors[i], label=f"KC {kc}")
    

axes[1].set_title("Courbes d'apprentissage — Top 6 KCs les plus fréquents")
axes[1].set_xlabel("Numéro de tentative sur le KC")
axes[1].set_ylabel("Taux de réussite moyen")
axes[1].set_ylim(0.4, 1.0)
axes[1].legend(title="KC", fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.show()"""


print("══════════════════ RÉSUMÉ ══════════════════")
print(f"Interactions totales     : {len(df):,}")
print(f"Élèves                   : {df['user_id'].nunique():,}")
print(f"KCs uniques              : {df['kc_main'].nunique()}")
print(f"Taux de réussite global  : {df['correct'].mean()*100:.1f}%")
print(f"Médiane inter./élève     : {inter_per_student.median():.0f}")
print(f"Élèves avec TR < 60%     : {(student_stats['taux_reussite'] < 0.6).sum()} "
      f"({(student_stats['taux_reussite'] < 0.6).mean()*100:.1f}%)")
print(f"KCs avec < 1000 inter.   : {(inter_per_kc < 1000).sum()}")
#choisir le kc le plus fréquenté pour faire les courbes d'apprentissage individuelles
kc_top = inter_per_kc.head(1).index[0]
print(f"KC le plus fréquenté : {kc_top}")
#5 user  les plus actifs sur ce KC
top5_users = df_kc[df_kc["kc"] == kc_top]["user_id"].value_counts().head(5).index.tolist()

fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 5))

for i, user in enumerate(top5_users):
    df_user = (df_kc[(df_kc["kc"] == kc_top) & (df_kc["user_id"] == user)]
               .sort_values("datetime"))

    # Taux de réussite glissant par semaine
    df_user = df_user.set_index("datetime")
    weekly = df_user["correct"].resample("W").mean().dropna()
    std= df_user["correct"].resample("W").std().dropna()
    se = std / np.sqrt(df_user["correct"].resample("W").count().dropna())
    ax.plot(weekly.index, weekly.values,
            marker="o", markersize=4, linewidth=2,
            color=colors[i], label=f"Élève {i+1}")
    
    

ax.set_title(f"Évolution hebdomadaire du taux de réussite\nKC : '{kc_top}'")
ax.set_xlabel("Date")
ax.set_ylabel("Taux de réussite (moyenne hebdomadaire)")
ax.set_ylim(0, 1.05)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="seuil 80%")
ax.legend(title="Élèves", loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()