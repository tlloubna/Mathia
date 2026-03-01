import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.graphics.PlotOutills as PO
import src.Analysis.DifficultyAnalyzer as DA
import src.Process.ForgettingModel as FGM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
DATA_FOLDER = os.path.join("data", "bridge_algebra06")
N_STUDENTS = 100 # Number of students to use
MIN_INTERACTIONS = 30
MODEL_C = 0.01  # Regularization parameter
N_TIME_WINDOWS = 5

def load_student_model(data_folder: str,mininteractions: int = 30,n_students: int = 100):
    """Load student data model to get metadata"""
    pathbridge = os.path.join(data_folder, "..", "bridge_algebra06", "data.txt")
    stdmodel :SD.StudentDATA = SD.StudentDATA(file=pathbridge, seed=42)
    df,Q=stdmodel.loadData(Display=False, min_intercation=mininteractions, n_students=n_students)
    
    return stdmodel,df,Q
def compute_forgetting_curves(df):
    
    df = df.sort_values(["user_id", "KC", "timestamp"])
    df["delta_t"] = np.nan
    last_time = {}
    for idx, row in df.iterrows():
        key = (row["user_id"], row["KC"])
        if key in last_time:
            df.at[idx, "delta_t"] = row["timestamp"] - last_time[key]
        last_time[key] = row["timestamp"]

    return df


def build_forgetting_curve(df):

    # enlever premières interactions
    df = df.dropna(subset=["delta_t"])
    bins = [
        0,
        3600,
        3600*24,
        3600*24*7,
        3600*24*30,
        df["delta_t"].max()
    ]
    bin_labels = ["<1h", "1h-1j", "1j-7j", "7j-30j", ">30j"]
    df["time_bin"] = pd.cut(df["delta_t"], bins=bins, labels=bin_labels, include_lowest=True)
    curve = df.groupby("time_bin")["correct"].mean()
    return curve

def plot(df):
    curve = build_forgetting_curve(df)
    plt.plot(curve.index.astype(str), curve.values, marker='o')
    plt.xticks(rotation=45, fontsize=18)
    plt.ylabel("Probabilité de réussite", fontsize=20)
    plt.xlabel("Temps depuis dernière révision", fontsize=20)
    plt.show()
def plot_forgetting_per_student_curves_only(df):
    # Étudiant avec le plus d'interactions
    top_student = df["user_id"].value_counts().idxmax()
    print(f"Étudiant avec le plus d'interactions : {top_student}")

    # Sélection des données de l'étudiant
    df_student = df[df["user_id"] == top_student].dropna(subset=["delta_t"])
    bins = [0, 3600, 3600*24, 3600*24*7, 3600*24*30, df_student["delta_t"].max()]
    bin_labels = ["<1h", "1h-1j", "1j-7j", "7j-30j", ">30j"]
    df_student["time_bin"] = pd.cut(df_student["delta_t"], bins=bins, labels=bin_labels, include_lowest=True)

    # Calcul des courbes par KC
    curves = df_student.groupby(["KC", "time_bin"])["correct"].mean().unstack(level=0)

    # Tracer uniquement les courbes
    plt.figure(figsize=(12,6), dpi=120)
    curves_clean = curves.loc[:, curves.isna().sum() <= 1]

    for kc in curves_clean.columns[:10]:
        plt.plot(curves_clean.index.astype(str), curves_clean[kc], marker='o', linewidth=2)
    plt.ylabel("Probabilité de réussite", fontsize=20)
    plt.xlabel("Temps depuis dernière révision", fontsize=20)
    plt.xticks(rotation=45, fontsize=18)
    plt.title(f"Courbes d'oubli par KC pour l'étudiant {top_student}", fontsize=20)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(curves_clean.columns[:10], title="KC", fontsize=12)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    stdmodel,df,Q = load_student_model(DATA_FOLDER, MIN_INTERACTIONS, N_STUDENTS)
    df = compute_forgetting_curves(df)
    plot(df)
    plot_forgetting_per_student_curves_only(df)
    print("Done!!!!!!!!!!!!!!!!")