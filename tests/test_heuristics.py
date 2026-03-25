import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import src.Process.Heuristics.Theta_TresholdH as TTH
import src.Process.DAS3H as das3H   
import src.datamodel.Studentdata as STD
import src.Process.Simulation.SimuH as SimuH
import src.Process.Heuristics.mu_back as MuBackH
import src.Process.Heuristics.Random as RandomH
import src.Process.Simulation.utils as simu_utils
import pandas as pd
from scipy import sparse
import joblib
import numpy as np 
import matplotlib.pyplot as plt
from utils.this_queue import OurQueue
import time
from collections import defaultdict
NAME_FOLDER="Mathiadata" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 25351
def load_Model():
    # Load the model
    df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    model_pathO = os.path.join(DATA_FOLDER, f"das3h_model_C0.1_{N_STUDENTS}std.pkl")
    kc_list=np.load(os.path.join(DATA_FOLDER,f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)["kc_list"]
    return df,q_matrix,model_pathO,kc_list

def ChooseStuden(df, method="Random"):
    if method=="Random":
        return df["user_id"].sample(1).iloc[0]
    elif method=="MostViewed":
        return df["user_id"].value_counts().idxmax()
    elif method=="MediumViewed":
       return df["user_id"].value_counts().index[len(df["user_id"].value_counts())//2]
    elif method=="LowerPerformance":
        performance = df.groupby("user_id")["correct"].mean()
        return performance.idxmin()
    elif method=="HigherPerformance":        
        performance = df.groupby("user_id")["correct"].mean()
        return performance.idxmax() 
    else:
        raise ValueError("Unknown method for choosing student")


"""def TestSimulation(nb_student,data,model,qmat,heuristic,seuil=0.5):
    simu=SimuH.SimulationH(data=data,model=model,qmat=qmat,heuristic=heuristic)
    print("Running simulation with review...")
    PMR_ap_rev, PMR_ret_rev = simu.Simulation(nb_students=nb_student, reviews_per_step=3, review=True)
    print("Running simulation without review...")
    PMR_ap_no, PMR_ret_no = simu.Simulation(nb_students=nb_student, reviews_per_step=3, review=False)

    weeks = list(range(16))
    plt.plot(weeks, np.concatenate([PMR_ap_rev, PMR_ret_rev]), label="theta_threshold", color="blue")
    plt.plot(weeks, np.concatenate([PMR_ap_no, PMR_ret_no]), label="no_review", color="red")
    plt.axvline(x=10, color='gray', linestyle='--', label="Fin apprentissage")
    plt.legend()
    plt.show()"""


def TestSimulation(nb_runs, nb_students, data, model, qmat, heuristics_dict, colors_dict,kc_list_arr):
    """
    heuristics_dict = {
        "theta_threshold": heuristic_theta,
        "mu_back_1": heuristic_mu1,
        "mu_back_2": heuristic_mu2,
        "random": heuristic_random,
        "no_review": None
    }
    colors_dict = {
        "theta_threshold": "blue",
        "mu_back_1": "green",
        "mu_back_2": "orange",
        "random": "purple",
        "no_review": "red"
    }
    """
    results = {name: {"ap": [], "ret": []} for name in heuristics_dict}

    for run in range(nb_runs):
        print(f"\nRun {run+1}/{nb_runs}")
        
        for name, heuristic in heuristics_dict.items():
            print(f"  Heuristique : {name}")
            simu = SimuH.SimulationH(data=data, model=model, qmat=qmat, heuristic=heuristic,kc_list=kc_list_arr)
            
            review = (heuristic is not None)  # no_review si heuristic=None
            PMR_ap, PMR_ret = simu.Simulation(nb_students=nb_students, review=review)
            
            results[name]["ap"].append(PMR_ap)
            results[name]["ret"].append(PMR_ret)

    # Plot
    weeks = list(range(16))
    plt.figure(figsize=(12, 6))
    
    for name, color in colors_dict.items():
        ap_all  = np.array(results[name]["ap"])
        ret_all = np.array(results[name]["ret"])
        
        mean = np.concatenate([np.mean(ap_all, axis=0), np.mean(ret_all, axis=0)])
        std  = np.concatenate([np.std(ap_all, axis=0),  np.std(ret_all, axis=0)])
        
        plt.plot(weeks, mean, label=name, color=color)
        plt.fill_between(weeks, mean - std, mean + std, alpha=0.2, color=color)
    
    plt.axvline(x=10, color='gray', linestyle='--', label="Fin apprentissage")
    plt.xlabel("Semaines")
    plt.ylabel("PMR")
    plt.title(f"Comparaison heuristiques — {nb_runs} runs x {nb_students} étudiants")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Afficher ACPL et ACPR pour chaque heuristique
    print("\n--- Résultats ---")
    for name in heuristics_dict:
        ap_all  = np.array(results[name]["ap"])
        ret_all = np.array(results[name]["ret"])
        acpl = np.mean(ap_all)
        acpr = np.mean(ret_all)
        print(f"{name:20s} | ACPL = {acpl:.4f} | ACPR = {acpr:.4f}")
    
    return results


if __name__ == "__main__":
    df,q_matrix,model_pathO,kc_list=load_Model()
    model=joblib.load(model_pathO)["model"]
    StudentModel=STD.StudentDATA(df,q_matrix)
    student=ChooseStuden(df,method="Random")
    kc_list=np.load(os.path.join(DATA_FOLDER,f"history_metadata_{N_STUDENTS}std.npz"), allow_pickle=True)["kc_list"]
    heuristic=TTH.ThetaTresholdH(model,StudentModel,student=None,data=df,qmat=q_matrix,kc_list=kc_list,effort=1)
    kc_list_arr = kc_list.tolist()

    heuristics_dict = {
        "theta_threshold" : TTH.ThetaTresholdH(model, StudentModel, student=None,
                                                data=df, qmat=q_matrix, 
                                                kc_list=kc_list_arr, effort=1),
        "mu_back_1"       : MuBackH.MuBackH(mu=1, qmat=q_matrix, kc_list=kc_list_arr),
        "mu_back_2"       : MuBackH.MuBackH(mu=2, qmat=q_matrix, kc_list=kc_list_arr),
        "random"          : RandomH.RandomH(qmat=q_matrix, kc_list=kc_list_arr),
        "no_review"       : None
    }

    colors_dict = {
        "theta_threshold" : "blue",
        "mu_back_1"       : "green",
        "mu_back_2"       : "orange",
        "random"          : "purple",
        "no_review"       : "red"
    }
    start_time=time.time()
    results = TestSimulation(nb_runs=10,nb_students=500,data=df,model=model,qmat=q_matrix,heuristics_dict=heuristics_dict,colors_dict=colors_dict,kc_list_arr=kc_list_arr)
    end_time=time.time()
    print(f"Simulation time: {end_time - start_time} seconds")
    print("!!!!!!!!!!!!!Done!!!!!!!!!")