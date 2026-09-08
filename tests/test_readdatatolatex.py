
import os
import sys

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)
import os
import numpy as np
import pandas as pd
from scipy import sparse

DATASETS = [
     ("Mathiadata_v2",      100000,  "mathia"),
    ("ASSISTments13_12", 15698, "assist12"),
    ("bridge_algebra06", 1146,  "bridge06"),
    ("algebra05",         574,  "algebra05"),
   
]

TIME_UNIT = 86400.0   # secondes -> jours. Mets 1.0 si timestamps deja en jours.


def dataset_stats(folder, n_students, label):
    data_folder = os.path.join("data", folder)
    df = pd.read_csv(os.path.join(data_folder, f"preprocessed_data_{n_students}std.csv"))
    q_mat = sparse.load_npz(
        os.path.join(data_folder, f"q_mat_{n_students}std.npz")
    ).toarray()

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_skills = q_mat.shape[1]
    n_inter = len(df)
    mean_correct = df["correct"].mean()

    # Skills per item : moyenne du nombre de CC par item, sur les items observes
    observed = df["item_id"].unique().astype(int)
    skills_per_item = q_mat[observed].sum(axis=1).mean()

    df = df.sort_values(["user_id", "timestamp"])
    t = df["timestamp"].to_numpy(dtype=float) / TIME_UNIT

    # Mean study period : (dernier - premier timestamp) par eleve, moyenne
    span = df.groupby("user_id")["timestamp"].agg(lambda s: (s.max() - s.min()) / TIME_UNIT)
    mean_period = span.mean()

    # Mean skill delay : delai entre deux interactions consecutives
    # du meme eleve sur la meme CC (une interaction peut porter plusieurs CC)
    item_to_kcs = {i: np.flatnonzero(q_mat[i]) for i in observed}
    users = df["user_id"].to_numpy()
    items = df["item_id"].to_numpy(dtype=int)

    last_seen = {}          # (user, kc) -> timestamp precedent
    delays = []
    for u, it, ts in zip(users, items, t):
        for kc in item_to_kcs[it]:
            key = (u, kc)
            if key in last_seen:
                delays.append(ts - last_seen[key])
            last_seen[key] = ts
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    return {
        "label": label,
        "users": n_users,
        "items": n_items,
        "skills": n_skills,
        "interactions": n_inter,
        "mean_correct": mean_correct,
        "skills_per_item": skills_per_item,
        "mean_skill_delay": mean_delay,
        "mean_study_period": mean_period,
    }


def to_latex_row(s):
    return (
        f"{s['label']} & {s['users']:,} & {s['items']:,} & {s['skills']:,} & "
        f"{s['interactions']:,} & {s['mean_correct']:.3f} & "
        f"{s['skills_per_item']:.3f} & {s['mean_skill_delay']:.2f} & "
        f"{s['mean_study_period']:.1f} \\\\"
    )


if __name__ == "__main__":
    rows = []
    for folder, n_students, label in DATASETS:
        try:
            s = dataset_stats(folder, n_students, label)
            rows.append(s)
            print(f"[ok] {label:<12} users={s['users']:>6} items={s['items']:>7} "
                  f"kc={s['skills']:>4} inter={s['interactions']:>9,}")
        except Exception as e:
            print(f"[!!] {label}: {e}")

    print("\n" + "=" * 70)
    print("\\begin{tabular}{@{}lrrrrrrrr@{}}")
    print("\\toprule")
    print("Dataset & Users & Items & Skills & Interactions & \\makecell[cl]{Mean \\\\ correctness} & ")
    print("\\makecell[cl]{Skills \\\\ per item} &")
    print("\\makecell[cl]{Mean \\\\ skill delay} &")
    print("\\makecell[cl]{Mean \\\\ study period} \\\\")
    print("\\midrule")
    for s in rows:
        print(to_latex_row(s))
    print("\\bottomrule")
    print("\\end{tabular}")