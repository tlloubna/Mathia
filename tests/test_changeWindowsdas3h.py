import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import src.datamodel.Historydata as HIS
import src.Process.DAS3H as DAS3H

NAME_FOLDER="Mathiadata" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS = 25351 # Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30


def test_changeWindowdas3h(windows,data,Q_mat):
    his=HIS.HistoryDATA(TimeWindow=windows)
    X,user_ids,item_ids,listofKC=his.ComputeHistoryFeaturesTWKC(Q_mat,data)
    model=DAS3H.DAS3HModel(C=1.0)
    results=model.fit(X,user_ids,item_ids,listofKC,n_tw=len(windows),perc_init=0.2)
    return results

def plot_window_comparaison(windows_configs, auc_list, nll_list, rmse_list):
    x=list(windows_configs.keys())
    fig,ax1=plt.subplots(figsize=(12,6))

    ax1.plot(x, auc_list, marker='o', label='AUC', color='blue')
    ax1.set_ylabel('AUC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2=ax1.twinx()
    ax2.plot(x, nll_list, marker='o', label='NLL', color='orange')
    ax2.set_ylabel('NLL', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')  
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax3=ax1.twinx()
    ax3.plot(x, rmse_list, marker='o', label='RMSE', color='green')
    ax3.set_ylabel('RMSE', color='green')
    ax3.tick_params(axis='y', labelcolor='green')   
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    lines1,labels1=ax1.get_legend_handles_labels()
    lines2,labels2=ax2.get_legend_handles_labels()
    lines3,labels3=ax3.get_legend_handles_labels()
    ax1.legend(lines1+lines2+lines3,labels1+labels2+labels3,loc='best',fontsize=12)
    plt.tight_layout()  
    plt.show()

if __name__ == "__main__":
    df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    
    H  = 3600
    D  = 3600 * 24
    W  = 3600 * 24 * 7
    M  = 3600 * 24 * 30
    INF = float("inf")

    window_configs = {
        "sans_tw":        [INF],                        
        "1h_inf":             [H, INF],
        "1j_inf":             [D, INF],
        "1sem_inf":           [W, INF],
        "1h_1j_inf":          [H, D, INF],
        "1j_1sem_inf":        [D, W, INF],
        "1sem_1moi_inf":     [W, M, INF],
        "1h_1j_1sem_inf":     [H, D, W, INF],
        "1j_1sem_1mois_inf":  [D, W, M, INF],
        "1h_1j_1sem_1m_inf":  [H, D, W, M, INF],          
    }
    auc_list=[]
    nll_list=[]
    rmse_list=[]
    for w in window_configs.values():
        print(f"Testing with windows: {w}")
        results = test_changeWindowdas3h(w, df, q_matrix)
        print(f"Results for windows {w}: {results}\n")  
        auc_list.append(results["AUC"])
        nll_list.append(results["NLL"])
        rmse_list.append(results["RMSE"])
        print(f"windows: {w}, AUC: {results['AUC']} \n ")

    plot_window_comparaison(window_configs, auc_list, nll_list, rmse_list)
    print("!!!!!!!!!!!!Done!!!!!!!!!!!!")
print("!!!!!!!done!!!!!!!!!!")

    




