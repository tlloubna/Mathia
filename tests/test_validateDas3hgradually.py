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
import joblib
import time
import matplotlib.pyplot as plt
import src.datamodel.CrossValidation as cv
import src.datamodel.Historydata as history
import src.Process.DAS3H as das3h
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
NAME_FOLDER="bridge_algebra06"
DATA_FOLDER=os.path.join("data",NAME_FOLDER)
N_STUDENTS = 1146 


def Test_LearningCurve(model:das3h.DAS3HModel, data, Qmat, nb_folds=5, method="strongest", 
                       percentages=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],his:history.HistoryDATA=None):
    
    
    all_users = sorted(data["user_id"].unique())
    all_items = sorted(data["item_id"].unique())
    
    results_by_percentage = {
        "percentages": percentages,
        "AUC":[],
        "NLL": [],
        "RMSE": [],
        "AUC_mean": [],
        "AUC_std": [],
        "NLL_mean": [],
        "NLL_std": [],
        "RMSE_mean": [],
        "RMSE_std": [],
        "n_students_train": [],
        "n_interactions_train": []
    }
    cross_valid=cv.CrossValid(data,NAME_FOLDER,nb_folds,0.2,DATA_FOLDER)
    for perc in percentages:
        print(f"POURCENTAGE D'ÉLÈVES : {int(perc*100)}%")
        fold_results = {
            "AUC": [],
            "NLL": [],
            "RMSE": []
        }
        
        for fold_id in range(nb_folds):
            #print(f"\n--- Fold {fold_id + 1}/{nb_folds} ---")
            df_train_full_ids, df_test_ids = cross_valid.getfold(fold_id, method)
            df_train_full=data.loc[df_train_full_ids]
            df_test=data.loc[df_test_ids]
            train_students = df_train_full["user_id"].unique()
            n_students_to_keep = max(1, int(len(train_students) * perc))
            np.random.seed(42 + fold_id) 
            selected_students = np.random.choice(
                train_students, 
                size=n_students_to_keep, 
                replace=False
            )
            df_train = df_train_full[df_train_full["user_id"].isin(selected_students)]
            
            print(f"Train: {len(selected_students)} élèves, {len(df_train)} interactions")
            print(f"Test:  {len(df_test['user_id'].unique())} élèves, {len(df_test)} interactions")
            
            
            X_train, _, _, _ = his.ComputeHistoryFeaturesTWKC(Q_mat=Qmat,df=df_train,vocab_users=all_users,
                                                                        vocab_items=all_items)
            
            
            X_test, _, _, _ = his.ComputeHistoryFeaturesTWKC( Q_mat=Qmat,df=df_test,vocab_users=all_users, 
                                                                      vocab_items=all_items)
            results = model.fit_with_split(X_train, X_test)
            
            fold_results["AUC"].append(results["AUC"])
            fold_results["NLL"].append(results["NLL"])
            fold_results["RMSE"].append(results["RMSE"])
            
            print(f"AUC: {results['AUC']:.4f}, NLL: {results['NLL']:.4f}, RMSE: {results['RMSE']:.4f}")
        
        auc_mean = np.mean(fold_results["AUC"])
        auc_std = np.std(fold_results["AUC"])
        nll_mean = np.mean(fold_results["NLL"])
        nll_std = np.std(fold_results["NLL"])
        rmse_mean = np.mean(fold_results["RMSE"])
        rmse_std = np.std(fold_results["RMSE"])

        results_by_percentage["AUC"].append(fold_results["AUC"])
        results_by_percentage["NLL"].append(fold_results["NLL"])
        results_by_percentage["RMSE"].append(fold_results["RMSE"])
        results_by_percentage["AUC_mean"].append(auc_mean)
        results_by_percentage["AUC_std"].append(auc_std)
        results_by_percentage["NLL_mean"].append(nll_mean)
        results_by_percentage["NLL_std"].append(nll_std)
        results_by_percentage["RMSE_mean"].append(rmse_mean)
        results_by_percentage["RMSE_std"].append(rmse_std)
        results_by_percentage["n_students_train"].append(n_students_to_keep)
        results_by_percentage["n_interactions_train"].append(len(df_train))
        joblib.dump(
                results_by_percentage,
                os.path.join(DATA_FOLDER, NAME_FOLDER, "learning_curve_results.joblib")
            )
        print(f"\nRÉSULTAT {int(perc*100)}% élèves:")
        print(f"  AUC:  {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"  NLL:  {nll_mean:.4f} ± {nll_std:.4f}")
        print(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    
    return results_by_percentage

def statistical_analysis_learning_curve(results_by_percentage, metric="AUC"):
    percentages = results_by_percentage["percentages"]
    values_by_fold = results_by_percentage[f"{metric}"]

    print(f"ANALYSE STATISTIQUE — {metric}")
    # H0: Toutes les moyennes sont égales (pas d'effet du pourcentage)
    # H1: Au moins une moyenne diffère
    
    anova_result = f_oneway(*values_by_fold)
    
    print(f"\n1. TEST ANOVA (One-Way)")
    print(f"   H0: Pas d'effet du pourcentage d'élèves sur {metric}")
    print(f"   Statistique F: {anova_result.statistic:.4f}")
    print(f"   p-value: {anova_result.pvalue:.4e}")
    
    if anova_result.pvalue < 0.05:
        print(f"    Résultat: SIGNIFICATIF (p < 0.05)")
        print(f"   → Il existe au moins une différence significative entre les pourcentages")
    else:
        print(f"   Résultat: NON SIGNIFICATIF (p ≥ 0.05)")
        print(f"   → Pas de différence significative détectée")
        return anova_result, None
    

    # Comparaisons pairwise pour identifier quelles paires diffèrent
    
    print(f"\n2. TESTS POST-HOC (Tukey HSD)")
    print(f"   Comparaisons pairwise entre tous les pourcentages:")
    print(f"   (Identifie quelles paires sont significativement différentes)\n")
    
    # Préparer les données pour Tukey HSD
    data_for_tukey = []
    group_labels = []
    
    for i, perc in enumerate(percentages):
        for value in values_by_fold[i]:
            data_for_tukey.append(value)
            group_labels.append(f"{int(perc*100)}%")
    
    # Effectuer Tukey HSD
    tukey_result = pairwise_tukeyhsd(
        endog=data_for_tukey,
        groups=group_labels,
        alpha=0.05
    )
    
    print(tukey_result)
    print(f"\n3. RÉSUMÉ DES DIFFÉRENCES SIGNIFICATIVES")
    print(f"   Paires où p < 0.05 (différence significative) :\n")
    
    tukey_df = pd.DataFrame(
        data=tukey_result.summary().data[1:],
        columns=tukey_result.summary().data[0]
    )
    
    significant_pairs = tukey_df[tukey_df['reject'] == True]
    
    if len(significant_pairs) == 0:
        print("   Aucune paire significativement différente détectée.")
    else:
        for idx, row in significant_pairs.iterrows():
            print(f"   {row['group1']} vs {row['group2']}: "
                  f"diff = {float(row['meandiff']):.4f}, p = {float(row['p-adj']):.4e}")
    
    return anova_result, tukey_result


def full_statistical_report(results_by_percentage):
    
    print("RAPPORT STATISTIQUE COMPLET — LEARNING CURVE")
    for metric in ["AUC", "NLL", "RMSE"]:
        anova_result, tukey_result = statistical_analysis_learning_curve(
            results_by_percentage, 
            metric=metric
        )
       
def print_summary(results_by_percentage):
    print(f"{'Pourcentage':<12} {'AUC':>10} {'NLL':>15} {'RMSE':>15}")
    print("-" * 55)
    for i in range(len(results_by_percentage["percentages"])):
        pp = results_by_percentage["percentages"][i] * 100
        pt_mean = results_by_percentage["AUC_mean"][i]
        pt_std = results_by_percentage["AUC_std"][i]
        nll_mean = results_by_percentage["NLL_mean"][i]
        nll_std = results_by_percentage["NLL_std"][i]
        rmse_mean = results_by_percentage["RMSE_mean"][i]
        rmse_std = results_by_percentage["RMSE_std"][i]
        print(f"{pp:>6.0f}%      {pt_mean:>6.3f} ± {pt_std:>5.3f}   {nll_mean:>6.3f} ± {nll_std:>5.3f}   {rmse_mean:>6.3f} ± {rmse_std:>5.3f}")
    
def plot_learning_curve(results_by_percentage):
    percentages = np.array(results_by_percentage["percentages"]) * 100
    
    auc_mean = results_by_percentage["AUC_mean"]
    auc_std = results_by_percentage["AUC_std"]
    nll_mean = results_by_percentage["NLL_mean"]
    nll_std = results_by_percentage["NLL_std"]
    rmse_mean = results_by_percentage["RMSE_mean"]
    rmse_std = results_by_percentage["RMSE_std"]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_auc = 'steelblue'
    ax1.set_xlabel('Pourcentage d\'élèves dans le train (%)', fontsize=12)
    ax1.set_ylabel('AUC', color=color_auc, fontsize=12)
    ax1.plot(percentages, auc_mean, marker='o', linewidth=2, 
             color=color_auc, label='AUC')
    ax1.fill_between(
        percentages, 
        np.array(auc_mean) - np.array(auc_std),
        np.array(auc_mean) + np.array(auc_std),
        alpha=0.2,
        color=color_auc
    )
    ax1.tick_params(axis='y', labelcolor=color_auc)
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    
    color_nll = 'coral'
    color_rmse = 'forestgreen'
    
    ax2.set_ylabel('NLL / RMSE', fontsize=12)
    
    # NLL
    line_nll = ax2.plot(percentages, nll_mean, marker='s', linewidth=2, 
                        color=color_nll, label='NLL', linestyle='--')
    ax2.fill_between(
        percentages, 
        np.array(nll_mean) - np.array(nll_std),
        np.array(nll_mean) + np.array(nll_std),
        alpha=0.2,
        color=color_nll
    )
    
    # RMSE
    line_rmse = ax2.plot(percentages, rmse_mean, marker='^', linewidth=2, 
                         color=color_rmse, label='RMSE', linestyle=':')
    ax2.fill_between(
        percentages, 
        np.array(rmse_mean) - np.array(rmse_std),
        np.array(rmse_mean) + np.array(rmse_std),
        alpha=0.2,
        color=color_rmse
    )
    
    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    plt.title('Courbe d\'apprentissage de DAS3H\nAUC (↑) vs NLL/RMSE (↓)',
              fontsize=14)
    plt.tight_layout()
    
   
    plt.show()
if __name__ == "__main__":
    time_execution=2
    if time_execution==1:
        data=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        Qmat = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        model=das3h.DAS3HModel()
        his=history.HistoryDATA()
        start=time.time()
        results_by_percentage = Test_LearningCurve(model, data, Qmat, nb_folds=5, method="strongest", his=his)
        end=    time.time()
        print(f"Execution time: {end - start:.2f} seconds")
    elif time_execution==2:
        results_path=os.path.join(DATA_FOLDER, NAME_FOLDER, "learning_curve_results.joblib")
        loaded = joblib.load(results_path)
        results_by_percentage = loaded
        full_statistical_report(results_by_percentage)
        plot_learning_curve(results_by_percentage)
        print_summary(results_by_percentage)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!done")

print("!!!!!!!!!!!!!Done!!!!!!!!!!!")