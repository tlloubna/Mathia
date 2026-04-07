import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import src.datamodel.CrossValidation as cv
import src.Process.DAS3H as das3h
import src.datamodel.Historydata as history
import pandas as pd 
import numpy as np 
from scipy import sparse
import joblib
import time 
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
    
NAME_FOLDER="Mathiadata"
DATA_FOLDER=os.path.join("data",NAME_FOLDER)
N_STUDENTS=25351
def test_cross_validation(data:pd.DataFrame,nb_folds=5,perc_init=0.2,):
    if NAME_FOLDER == "Mathiadata":
        data=data.rename(columns={
            "student_id": "user_id",
            "kc_ids":     "KC",
        })

    cross_valid=cv.CrossValid(data,NAME_FOLDER,nb_folds,perc_init,DATA_FOLDER)
    cross_valid.saveStrongestFolds()
    cross_valid.savePseudoStrongFolds()
    print("Cross-validation folds created and verified successfully.")

def Test_His(nb_folds,data,Qmat,perc_init,his:history.HistoryDATA=None,):
    cross_valid=cv.CrossValid(data,NAME_FOLDER,nb_folds,perc_init,DATA_FOLDER)
    for fold_id in range(nb_folds):
        print(f"Processing fold {fold_id + 1}/{nb_folds}...")
        train_ids_strong,test_ids_strong=cross_valid.getfold(fold_id,"strongest")
        #train_ids_pseudo,test_ids_pseudo=cross_valid.getfold(fold_id,"pseudo_strong")
        df_train=data.loc[train_ids_strong]
        df_test=data.loc[test_ids_strong]

        all_users=sorted(data['user_id'].unique())
        all_items=sorted(data['item_id'].unique())
        X_train,user_ids,item_ids,listOfKC=his.ComputeHistoryFeaturesTWKC(Q_mat=Qmat,df=df_train,vocab_users=all_users,vocab_items=all_items)
        X_test,_,_,_=his.ComputeHistoryFeaturesTWKC(Q_mat=Qmat,df=df_test,vocab_users=all_users,vocab_items=all_items)
        
        
        sparse.save_npz(os.path.join(DATA_FOLDER,NAME_FOLDER,"strongest","folds",f"X_train_fold_{fold_id}.npz"),sparse.csr_matrix(X_train))
        sparse.save_npz(os.path.join(DATA_FOLDER,NAME_FOLDER,"strongest","folds",f"X_test_fold_{fold_id}.npz"),sparse.csr_matrix(X_test))
        np.savez(os.path.join(DATA_FOLDER,NAME_FOLDER,"strongest","folds",f"history_metadata_{N_STUDENTS}std_fold_{fold_id}.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listOfKC)
    
    print("Data saved for all folds.")

def Test_His_Pseudo(nb_folds,data,Qmat,perc_init,his:history.HistoryDATA=None,):
    cross_valid=cv.CrossValid(data,NAME_FOLDER,nb_folds,perc_init,DATA_FOLDER)
    #A revoir car j'ai enlver pour traier le dossier 4, je l'ai pas traite avant je ne sais pourquoi :-)
    print(f"Processing fold {4 + 1}/{nb_folds}...")
    train_ids_pseudo,test_ids_pseudo=cross_valid.getfold(4,"pseudo_strong")
    df_train=data.loc[train_ids_pseudo]
    df_test=data.loc[test_ids_pseudo]

    all_users=sorted(data['user_id'].unique())
    all_items=sorted(data['item_id'].unique())
    X_train,user_ids,item_ids,listOfKC=his.ComputeHistoryFeaturesTWKC(Q_mat=Qmat,df=df_train,vocab_users=all_users,vocab_items=all_items)
    X_test,_,_,_=his.ComputeHistoryFeaturesTWKC(Q_mat=Qmat,df=df_test,vocab_users=all_users,vocab_items=all_items)

    sparse.save_npz(os.path.join(DATA_FOLDER,NAME_FOLDER,"pseudo_strong","folds",f"X_train_fold_{4}.npz"),sparse.csr_matrix(X_train))
    sparse.save_npz(os.path.join(DATA_FOLDER,NAME_FOLDER,"pseudo_strong","folds",f"X_test_fold_{4}.npz"),sparse.csr_matrix(X_test))
    np.savez(os.path.join(DATA_FOLDER,NAME_FOLDER,"pseudo_strong","folds",f"history_metadata_{N_STUDENTS}std_fold_{4}.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listOfKC)

    print("Data saved for all pseudo-strong folds.")


def Test_model(nb_folds, model: das3h.DAS3HModel = None, name_dataset=NAME_FOLDER, method="strongest"):
    all_results = {
        "AUC": [],
        "NLL": [],
        "RMSE": [],
        "y_test": [],
        "y_pred": [],
        "FPR": [],
        "TPR": []
    }
    
    for fold_id in range(nb_folds):
        
        print(f"FOLD {fold_id + 1}/{nb_folds}")
        X_train = sparse.load_npz(
            os.path.join(DATA_FOLDER, name_dataset, method, "folds", f"X_train_fold_{fold_id}.npz")
        )
        X_test = sparse.load_npz(
            os.path.join(DATA_FOLDER, name_dataset, method, "folds", f"X_test_fold_{fold_id}.npz")
        )
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        fold_results = model.fit_with_split(X_train, X_test)
        
        all_results["AUC"].append(fold_results["AUC"])
        all_results["NLL"].append(fold_results["NLL"])
        all_results["RMSE"].append(fold_results["RMSE"])
        all_results["y_test"].append(fold_results["y_test"])
        all_results["y_pred"].append(fold_results["y_pred"])
        all_results["FPR"].append(fold_results["FPR"])
        all_results["TPR"].append(fold_results["TPR"])
        
        print(f"Fold {fold_id + 1} - AUC: {fold_results['AUC']:.4f}, "
              f"NLL: {fold_results['NLL']:.4f}, "
              f"RMSE: {fold_results['RMSE']:.4f}")
        
        joblib.dump(
            {"model": model, "results": fold_results},
            os.path.join(DATA_FOLDER, name_dataset, method, "folds", f"model_fold_{fold_id}.joblib")
        )
    
    print(f"RÉSULTATS FINAUX - {method.upper()}")
    
    summary = {}
    for metric in ["AUC", "NLL", "RMSE"]:
        mean = np.mean(all_results[metric])
        std = np.std(all_results[metric])
        summary[metric] = {"mean": mean, "std": std}
        print(f"{metric:6s}: {mean:.4f} ± {std:.4f}")
    
    joblib.dump(
        {"summary": summary, "all_results": all_results},
        os.path.join(DATA_FOLDER, name_dataset, method, "cross_validation_results.joblib")
    )
    
    return summary, all_results

def print_summary(all_results, method="strongest",nb_folds=5):
    print(f"\nRÉSULTATS FINAUX - {method.upper()}")
    
    for fold in range(nb_folds):
        print(f"Fold {fold + 1} - AUC: {all_results['AUC'][fold]:.4f}, "
              f"NLL: {all_results['NLL'][fold]:.4f}, "
              f"RMSE: {all_results['RMSE'][fold]:.4f}")
    
    summary={}
    for metric in ["AUC", "NLL", "RMSE"]:
        mean = np.mean(all_results[metric])
        std = np.std(all_results[metric])
        summary[metric] = {"mean": mean, "std": std}
        print(f"{metric:6s}: {mean:.4f} ± {std:.4f}")   
    return summary

def plot_calibration_complete(all_results, method="strongest"):
    nb_folds = len(all_results["y_test"])
    y_true_all = np.concatenate(all_results["y_test"])
    y_pred_all = np.concatenate(all_results["y_pred"])
    prob_true_global, prob_pred_global = calibration_curve(y_true_all, y_pred_all, n_bins=10)
    prob_true_per_fold = []
    prob_pred_per_fold = []
    for fold_id in range(nb_folds):
        y_true = all_results["y_test"][fold_id]
        y_pred = all_results["y_pred"][fold_id]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        prob_true_per_fold.append(prob_true)
        prob_pred_per_fold.append(prob_pred)
    prob_true_per_fold = np.array(prob_true_per_fold)  
    prob_true_mean = np.mean(prob_true_per_fold, axis=0)
    prob_true_std = np.std(prob_true_per_fold, axis=0)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(prob_pred_global, prob_true_global,
             marker='o', markersize=8, color="steelblue",
             linewidth=2, label="DAS3H (global)", zorder=3)
    
    ax1.plot([0, 1], [0, 1],
             linestyle="--", color="gray", linewidth=1.5,
             label="Calibration parfaite", zorder=2)
    
    ax1.fill_between(prob_pred_global,
                     prob_true_mean - prob_true_std,
                     prob_true_mean + prob_true_std,
                     alpha=0.3, color="steelblue",
                     label="± std (entre folds)", zorder=1)
    
    ax1.set_xlabel("Probabilité prédite", fontsize=11)
    ax1.set_ylabel("Taux réel de réussite", fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Calibration globale\n(tous les folds)", fontsize=12)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, nb_folds))
    
    for fold_id in range(nb_folds):
        ax2.plot(prob_pred_per_fold[fold_id], prob_true_per_fold[fold_id],
                 marker='o', markersize=5, color=colors[fold_id],
                 linewidth=1.5, alpha=0.7,
                 label=f"Fold {fold_id + 1}")
    
    ax2.plot([0, 1], [0, 1],
             linestyle="--", color="gray", linewidth=2,
             label="Parfaite")
    
    ax2.set_xlabel("Probabilité prédite", fontsize=11)
    ax2.set_ylabel("Taux réel de réussite", fontsize=11)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Calibration par fold\n(variabilité)", fontsize=12)
    plt.suptitle(f"Calibration du modèle DAS3H — {method.capitalize()}\n"
                 f"Validation croisée ({nb_folds} folds, n={len(y_true_all)} prédictions)",
                 fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\nCalibration globale — {method.capitalize()}")
    print(f"{'Bin':<5} {'Prob prédite':<15} {'Taux réel (global)':<20} {'Taux réel (mean±std)':<25}")
    for i, (pp, pt_glob, pt_mean, pt_std) in enumerate(zip(
        prob_pred_global, prob_true_global, prob_true_mean, prob_true_std
    )):
        print(f"{i+1:<5} {pp:>6.3f}          {pt_glob:>6.3f}               {pt_mean:>6.3f} ± {pt_std:>5.3f}")

if __name__ == "__main__":
    time_execution=5
    if time_execution==1:
        data=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        start=time.time()
        test_cross_validation(data,nb_folds=5,perc_init=0.2)
        end=time.time()
        print(f"Execution time: {end - start:.2f} seconds")
    elif time_execution==2:
        data=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        Qmat=sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        start=time.time()
        Test_His(nb_folds=5,data=data,Qmat=Qmat,perc_init=0.2,his=history.HistoryDATA())
        end=time.time()
        print(f"Execution time: {end - start:.2f} seconds")

    elif time_execution==3:
        data=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        Qmat=sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        start=time.time()
        Test_His_Pseudo(nb_folds=5,data=data,Qmat=Qmat,perc_init=0.2,his=history.HistoryDATA())
        end=time.time()
        print(f"Execution time: {end - start:.2f} seconds")
    elif time_execution==4:
        nb_folds=5
        model=das3h.DAS3HModel()
        start=time.time()
        Test_model(nb_folds, model=model, name_dataset=NAME_FOLDER, method="pseudo_strong") 
        end=time.time()
        print(f"Execution time: {end - start:.2f} seconds")

    elif time_execution==5:
        results_path=os.path.join(DATA_FOLDER, NAME_FOLDER, "strongest", "cross_validation_results.joblib")
        loaded = joblib.load(results_path)
        all_results = loaded["all_results"]
        plot_calibration_complete(all_results, method="strongest")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!done")

    elif time_execution==6:
        results_path=os.path.join(DATA_FOLDER, NAME_FOLDER, "strongest", "cross_validation_results.joblib")
        loaded = joblib.load(results_path)
        all_results = loaded["all_results"]
        print_summary(all_results, method="pseudo_strong", nb_folds=5)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!done")

        
    print("Done ")




