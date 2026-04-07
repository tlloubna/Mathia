import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import os, sys
import numpy as np

import src.Process.DAS3H as DAS3H
import src.datamodel.Historydata as HIS
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt 
from collections import defaultdict
import random
NAME_FOLDER  = "bridge_algebra06"
DATA_FOLDER  = os.path.join("data", NAME_FOLDER)
N_STUDENTS   = 1146
MAX_GAP_DAYS = 7
MAX_SESSIONS = 10  # on trace jusqu'à 10 séances
def chooseStudenatleast90days(df, min_days=90):
    student_days = df.groupby("user_id")["timestamp"].agg(['min', 'max'])
    student_days['days_active'] = (student_days['max'] - student_days['min']) / (3600 * 24)
    return student_days[student_days['days_active'] >= min_days].index.tolist()

def chooseStudentsActive3days(df, max_gap=3, min_days=10):
    result = []
    for student, grp in df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        t_min = grp["timestamp"].min()
        days = sorted(((grp["timestamp"] - t_min) // (3600 * 24)).unique())
        
        if len(days) < min_days:  
            continue
        
        gaps = [days[i+1] - days[i] for i in range(len(days) - 1)]
        
        if max(gaps) <= max_gap:
            result.append(student)
    
    return result
def ChooseSplit(Choose=1, student=0, df=None):
    df_student = df[df["user_id"] == student].sort_values("timestamp")
    t_min = df_student["timestamp"].min()
    df_student = df_student.copy()
    df_student["day"] = (df_student["timestamp"] - t_min) // (3600 * 24)
    max_day = int(df_student["day"].max())
    df_Train, df_Test = [], []
    if Choose == 1:
        #les courbes ne sont pas stables parce qu'on change la base de test à chaque fois 
        #le modèle peut ne pas voir les exercices dans la phase d'apprentissage donc on ne sait pas
        #la difficulte de chaque exercice et aussi ces compétences.
        df_student = df_student.sort_values("day")
        for i in range(1, max_day + 1):
            train = df_student[df_student["day"] < i]
            test  = df_student[df_student["day"] == i]
            if len(train) == 0 or len(test) == 0:
                continue
            df_Train.append(train)
            df_Test.append(test)
    elif Choose == 2:
        active_days = sorted(df_student["day"].unique())
        
        for i in range(len(active_days) - 1):
            current_day = active_days[i]
            next_day    = active_days[i + 1]
            
            # contrainte gap <= 7 jours
            if next_day - current_day > MAX_GAP_DAYS:
                continue
            
            train = df_student[df_student["day"] <= current_day]
            test  = df_student[df_student["day"] == next_day]
            
            if len(train) == 0 or len(test) == 0:
                continue
            
            df_Train.append(train)
            df_Test.append(test)

    elif Choose == 3:
        n_lots = 5
        df_student = df_student.sort_values("timestamp")
        n = len(df_student)
        lot_size = n // n_lots
        if lot_size == 0:
            return df_Train, df_Test
        
        lots = []
        for k in range(n_lots):
            start = k * lot_size
            end   = (k + 1) * lot_size if k < n_lots - 1 else n  # dernier lot prend le reste
            lots.append(df_student.iloc[start:end])
        for k in range(len(lots) - 1):
            train = pd.concat(lots[:k+1])   # lots 0..k
            test  = lots[k+1]               # lot suivant
            
            if len(train) == 0 or len(test) == 0:
                continue
            
            df_Train.append(train)
            df_Test.append(test)
    return df_Train, df_Test

def SplitPerLot(nb_student=10, df=None):
    students = chooseStudenatleast90days(df)
    #random.shuffle(students)
    selected = students[:nb_student]
    
    df_lot = df[df["user_id"].isin(selected)].copy()
    t_min = df_lot["timestamp"].min()
    df_lot["day"] = (df_lot["timestamp"] - t_min) // (3600 * 24)
    max_day = int(df_lot["day"].max())
    
    df_train, df_test = [], []
    
    for t in range(10, max_day + 1, 10):
        split_df = df_lot[df_lot["day"] < t].sort_values("timestamp")
        n        = len(split_df)
        split    = max(1, int(n * 0.8))
        
        train = split_df.iloc[:split]
        test  = split_df.iloc[split:]
        
        if len(train) == 0 or len(test) == 0:
            continue
            
        df_train.append(train)
        df_test.append(test)
    
    return df_train, df_test


def test_das3hchrono(df_train, df_test, q_matrix, windows, all_users, all_items):
    his = HIS.HistoryDATA(TimeWindow=windows)
    
    X_train, user_train, item_train, kc_train = his.ComputeHistoryFeaturesTWKC(
        q_matrix, df_train,
        vocab_users=all_users, vocab_items=all_items
    )
    X_test, user_test, item_test, kc_test = his.ComputeHistoryFeaturesTWKC(
        q_matrix, df_test,
        vocab_users=all_users, vocab_items=all_items
    )
    kc_list = [str(i) for i in range(q_matrix.shape[1])]  
    
    model = DAS3H.DAS3HModel(C=0.01)
    results = model.fit_with_split(
        X_train, X_test,
        n_user_ids=user_train,   # len = len(all_users)
        item_ids=item_train,     # len = len(all_items)
        kc_list=kc_list,         # len = Q_mat.shape[1]
        n_tw=len(windows)
    )
    
    params = model.get_params()
    n_params = X_train.shape[1] - 1  # -1 pour colonne correct
    n_inter  = X_train.shape[0]
    print(f"n_params={n_params}, n_inter={n_inter}, ratio={n_params/n_inter:.2f}")
    return results, params

def testifTrainTesthas2class(df_train,df_test):
    if len(df_train["correct"].unique())==1 or len(df_test["correct"].unique())==1:
        return False
    return True
def plot_comparaison(results_list: dict, choose=1, choose_split=3, n_lots=5):
    if choose == 1:
        title_1 = "Method 1: Par étudiant"
    elif choose == 2:
        title_1 = "Method 2: Par lot d'étudiants"

    if choose_split == 1:
        title_2 = "Split : chaque jour => train =< day, test = day"
    elif choose_split == 2:
        title_2 = "Split 2:  train =< day, test = day+1 avec gap <= 3 jours"
    elif choose_split == 3:
        title_2 = "Split 3: Train in activate days : train =< day_a, test = day_a+1"
    elif choose_split == 4:
        title_2 = f"Split 4: {n_lots} lots d'interactions — train cumulatif, test = lot suivant"

    students = list(results_list.keys())
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax_auc, ax_nll, ax_rmse = axes

    for student in students:
        res_list = results_list[student]
        if len(res_list) == 0:
            continue
        days      = list(range(1, len(res_list) + 1))
        auc_list  = [r["AUC"]  for r in res_list]
        nll_list  = [r["NLL"]  for r in res_list]
        rmse_list = [r["RMSE"] for r in res_list]

        ax_auc.plot(days,  auc_list,  linewidth=0.8, alpha=0.4, color="steelblue")
        ax_nll.plot(days,  nll_list,  linewidth=0.8, alpha=0.4, color="coral")
        ax_rmse.plot(days, rmse_list, linewidth=0.8, alpha=0.4, color="forestgreen")

    auc_by_day  = defaultdict(list)
    nll_by_day  = defaultdict(list)
    rmse_by_day = defaultdict(list)
    for res_list in results_list.values():
        for d, r in enumerate(res_list):
            auc_by_day[d].append(r["AUC"])
            nll_by_day[d].append(r["NLL"])
            rmse_by_day[d].append(r["RMSE"])

    days_sorted = sorted(auc_by_day.keys())
    x_ticks     = list(range(1, len(days_sorted) + 1))

    ax_auc.plot(x_ticks, [np.mean(auc_by_day[d])  for d in days_sorted],
                color="steelblue", linewidth=2, label="Moyenne")
    ax_auc.fill_between(x_ticks,
                        [np.mean(auc_by_day[d]) - np.std(auc_by_day[d]) for d in days_sorted],
                        [np.mean(auc_by_day[d]) + np.std(auc_by_day[d]) for d in days_sorted],
                        color="steelblue", alpha=0.1)

    ax_nll.plot(x_ticks, [np.mean(nll_by_day[d])  for d in days_sorted],
                color="coral", linewidth=2)
    ax_nll.fill_between(x_ticks,
                        [np.mean(nll_by_day[d]) - np.std(nll_by_day[d]) for d in days_sorted],
                        [np.mean(nll_by_day[d]) + np.std(nll_by_day[d]) for d in days_sorted],
                        color="coral", alpha=0.1)

    ax_rmse.plot(x_ticks, [np.mean(rmse_by_day[d]) for d in days_sorted],
                 color="forestgreen", linewidth=2)
    ax_rmse.fill_between(x_ticks,
                         [np.mean(rmse_by_day[d]) - np.std(rmse_by_day[d]) for d in days_sorted],
                         [np.mean(rmse_by_day[d]) + np.std(rmse_by_day[d]) for d in days_sorted],
                         color="forestgreen", alpha=0.1)

    # labels de l'axe X
    if choose_split == 4:
        lot_labels = [f"Lots 1..{k}\n({int(k/n_lots*100)}%)" for k in range(1, n_lots)]
        tick_labels = lot_labels[:len(days_sorted)]
    else:
        tick_labels = [str(d+1) for d in days_sorted]

    for ax in [ax_auc, ax_nll, ax_rmse]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(tick_labels, fontsize=9)

    ax_auc.set_ylabel("AUC");  ax_auc.axhline(0.5, linestyle="--", color="gray")
    ax_nll.set_ylabel("NLL")
    ax_rmse.set_ylabel("RMSE")
    ax_auc.set_xlabel("Séance active ")

    ax_auc.legend(); ax_auc.grid(True, alpha=0.3)
    ax_nll.grid(True, alpha=0.3)
    ax_rmse.grid(True, alpha=0.3)
    plt.suptitle(f"Stabilité de DAS3H — {title_1}\n{title_2}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_comparaison_lots(results_list: dict, choose=1):
    colors = ["steelblue", "coral", "forestgreen", "purple", "orange"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    if choose == 1:
        title_1 = "Method 1: Par étudiant"
    elif choose == 2:
        title_1 = "Method 2: Par lot d'étudiants"

    for idx, (lot, res_list) in enumerate(results_list.items()):
        if len(res_list) == 0:
            continue
        color = colors[idx % len(colors)]
        steps = list(range(10, 10 * len(res_list) + 1, 10))
        auc_list  = [r["AUC"]  for r in res_list]
        nll_list  = [r["NLL"]  for r in res_list]
        rmse_list = [r["RMSE"] for r in res_list]

        ax1.plot(steps, auc_list,  marker="o", markersize=3, linewidth=1.2,
                 color=color, label=f"{lot} étudiants")
        ax2.plot(steps, nll_list,  marker="o", markersize=3, linewidth=1.2,
                 color=color)
        ax3.plot(steps, rmse_list, marker="o", markersize=3, linewidth=1.2,
                 color=color)

    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax1.set_ylabel("AUC ↑");  ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("NLL ↓");  ax2.grid(True, alpha=0.3)
    ax3.set_ylabel("RMSE ↓"); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Numéro de la séance active (Not days )")
    
    plt.suptitle(f"Performance de DAS3H par taille de lot d'étudiants — {title_1}", fontsize=13)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
    q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
    
    windows = [3600, 3600 * 24, 3600 * 24 * 7, 3600 * 24 * 30,float("inf")]
    students = chooseStudentsActive3days(df, max_gap=10, min_days=10)
    #random.shuffle(students)
    results_list = {}
    list_params={}
    all_users = sorted(df["user_id"].unique())
    all_items = sorted(df["item_id"].unique())
    choose=2
    max_student=30
    choose_split=2
    #sanity_check(df, q_matrix, windows)
    if choose ==1:
        for student in students[:10]:
            df_train, df_test = ChooseSplit(Choose=choose_split, student=student, df=df)
            list_res=[]
            for i, (df_tr, df_te) in enumerate(zip(df_train, df_test)):
                print(f"Step {i+1}/{len(df_train)} for student {student}")
                print(f"step {students.index(student)+1}/ 10 students")
                if testifTrainTesthas2class(df_tr, df_te):
                    results,params = test_das3hchrono(df_tr, df_te, q_matrix, windows, all_users, all_items)
                    list_res.append(results)
                    list_params[student] = params
                    print("stop")
                else:
                    print(f"Skipping student {student} at day {i} due to lack of class diversity.")
            results_list[student] = list_res
    elif choose ==2:
        for n in [10,20,30,40,50,100,200,300,400,500,600]:
            df_train, df_test = SplitPerLot(nb_student=n, df=df)
            list_res=[]
            for i, (df_tr, df_te) in enumerate(zip(df_train, df_test)):

                print(f"Step {i+1}/{len(df_train)} for lot of {n} students")
                if testifTrainTesthas2class(df_tr, df_te):
                    results, params = test_das3hchrono(df_tr, df_te, q_matrix, windows, all_users, all_items)
                    list_res.append(results)
                    list_params[n] = params
                else:
                    print(f"Skipping lot of {n} students at step {i} due to lack of class diversity.")
            results_list[n] = list_res

    
    plot_comparaison(results_list,choose = choose,choose_split=choose_split)
    plot_comparaison_lots(results_list,choose=choose)
print("!!!!!!!!!!!!Done!!!!!!!!!!!!")
   