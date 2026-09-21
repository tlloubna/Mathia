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
import src.datamodel.Studentdata as SD
import joblib
NAME_FOLDER="simulated"#algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)

def test_changeWindowdas3h(windows, data, Q_mat, nom):
    his = HIS.HistoryDATA(TimeWindow=windows)
    X, user_ids, item_ids, listofKC = his.ComputeHistoryFeaturesTWKC(Q_mat, data)
    sparse.save_npz(os.path.join(DATA_FOLDER, f"history_features_{nom}.npz"),
                    sparse.csr_matrix(X))
    np.savez(os.path.join(DATA_FOLDER, f"history_metadata_{nom}.npz"),
             user_ids=user_ids, item_ids=item_ids, kc_list=listofKC)
    model = DAS3H.DAS3HModel(C=1.0)
    results = model.fit(X, user_ids, item_ids, listofKC,
                        n_tw=len(windows), perc_init=0.8)
    out_path = os.path.join(DATA_FOLDER, f"das3h_model_C1_{nom}.pkl")
    joblib.dump({"mode": "production", "model": model, "results": results}, out_path)
    return results

def load_student_model(data_folder: str,mininteractions: int = 30,n_students: int = 100):
    print("!!!!!!!!!!!!!!!!Loading student model !!!!!!!!!!!!")
    pathbridge = os.path.join(data_folder, "..", NAME_FOLDER, "data.csv")
    stdmodel :SD.StudentDATA= SD.StudentDATA(file=pathbridge)
    df,Q=stdmodel.loadData(Display=False, min_intercation=mininteractions, n_students=n_students)
    df.to_csv(os.path.join(data_folder, f"preprocessed_data.csv"), index=False)
    sparse.save_npz(os.path.join(data_folder, f"q_mat.npz"), sparse.csr_matrix(Q))

    joblib.dump({"user_mapping": stdmodel.user_mapping,
                 "item_mapping": stdmodel.item_mapping,},
                os.path.join(data_folder, f"id_mappings.pkl"))
    return df,Q



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
    df=pd.read_csv("/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/preprocessed_data_simulated_1000std.csv")
    q_matrix=sparse.load_npz("/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/q_mat_1000std.npz").toarray()
    #df,q_matrix=load_student_model(data_folder=DATA_FOLDER,mininteractions=1,n_students=100000)
    H    = 3600
    D    = 3600 * 24
    W    = 3600 * 24 * 7
    M    = 3600 * 24 * 30
    T_M  = 3600 * 24 * 30 * 3       # 3 mois
    S_M  = 3600 * 24 * 30 * 6       # 6 mois
    O_Y  = 3600 * 24 * 30 * 12      # 1 an
    INF  = float("inf")

    window_configs = {
        # references deja mesurees
        "das3h_original":        [H, D, W, M, INF],
        "w_m_3m_6m_1an_inf":     [W, M, T_M, S_M, O_Y, INF],
        "m_3m_6m_1an_inf":       [M, T_M, S_M, O_Y, INF],
        "3m_6m_1an_inf":         [T_M, S_M, O_Y, INF],
        "6m_1an_inf":            [S_M, O_Y, INF],
    }
    auc_list=[]
    nll_list=[]
    rmse_list=[]
    for nom, w in window_configs.items():
        print(f"Testing with windows: {w}")
        results =test_changeWindowdas3h(w, df, q_matrix, nom)
        auc_list.append(results["AUC"])
        nll_list.append(results["NLL"])
        rmse_list.append(results["RMSE"])
        print(f"windows: {w}, AUC: {results['AUC']} \n ")
        print(f"windows: {w}, NLL: {results['NLL']} \n ")
        print(f"windows: {w}, RMSE: {results['RMSE']} \n ")


    plot_window_comparaison(window_configs, auc_list, nll_list, rmse_list)
    print("!!!!!!!!!!!!Done!!!!!!!!!!!!")
print("!!!!!!!done!!!!!!!!!!")

    




