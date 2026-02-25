

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

import src.datamodel.Studentdata as SD
import matplotlib.pyplot as plt
import src.Process.DAS3H as DAS3H
import src.datamodel.Historydata as HIS
import src.graphics.PlotOutills as Plot
import src.graphics.das3hviz as Vis


# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_FOLDER = os.path.join("data", "bridge_algebra06")
N_STUDENTS = 1000 # Number of students to use
MIN_INTERACTIONS = 30
MODEL_C = 0.01  # Regularization parameter
N_TIME_WINDOWS = 5


def setup_data(data_folder: str, n_students: int = 100):
    """Load preprocessed data and Q matrix"""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    csv_path = os.path.join(data_folder, f"preprocessed_data_{n_students}std.csv")
    q_matrix_path = os.path.join(data_folder, f"q_mat_{n_students}std.npz")
    
    df = pd.read_csv(csv_path, sep=",")
    q_matrix = sparse.load_npz(q_matrix_path).toarray()
    
    print(f"✓ Data shape: {df.shape}")
    print(f"✓ Q-matrix shape: {q_matrix.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    return df, q_matrix


def load_student_model(data_folder: str,mininteractions: int = 30,n_students: int = 100):
    """Load student data model to get metadata"""
    pathbridge = os.path.join(data_folder, "..", "bridge_algebra06", "data.txt")
    stdmodel :SD.StudentDATA = SD.StudentDATA(file=pathbridge)
    df,Q=stdmodel.loadData(Display=False, min_intercation=mininteractions, n_students=n_students)
    
    return stdmodel,df,Q


def prepare_features(df: pd.DataFrame, q_matrix: np.ndarray, stdmodel: SD.StudentDATA):
    """Compute DAS3H history features"""
    print(f"\n{'='*70}")
    print("COMPUTING HISTORY FEATURES")
    print(f"{'='*70}")
    
    his = HIS.HistoryDATA(stdmodel=stdmodel)
    X = his.ComputeHistoryFeaturesTWKC(Q_mat=q_matrix, df=df)
    
    print(f"✓ Feature matrix X shape: {X.shape}")
    print(f"✓ Number of users: {len(his.user_ids)}")
    print(f"✓ Number of items: {len(his.item_ids)}")
    print(f"✓ Number of skills: {len(stdmodel.KComp)}")
    
    return X, his.user_ids, his.item_ids, stdmodel.KComp


def test_model_training(X, user_ids, item_ids, kc_list, model_c: float = 0.1, n_tw: int = 5):
    """Train DAS3H model and test results"""
    print(f"\n{'='*70}")
    print("TRAINING DAS3H MODEL")
    print(f"{'='*70}")
    print(f"Model parameters: C={model_c}, n_time_windows={n_tw}")
    
    model = DAS3H.DAS3HModel(C=model_c)
    results = model.fit(
        X,
        user_ids=user_ids,
        item_ids=item_ids,
        kc_list=kc_list,
        n_tw=n_tw
    )
    
    print(f"\n{'─'*70}")
    print("TRAINING RESULTS")
    print(f"{'─'*70}")
    print(f"✓ AUC:  {results['AUC']:.4f}")
    print(f"✓ ACC:  {results['ACC']:.4f}")
    print(f"✓ NLL:  {results['NLL']:.4f}")
    print(f"✓ RMSE: {results['RMSE']:.4f}")
    
    return model, results


def test_model_parameters(model: DAS3H.DAS3HModel):
    """Extract and test model parameters"""
    print(f"\n{'='*70}")
    print("TESTING MODEL PARAMETERS")
    print(f"{'='*70}")
    
    params = model.get_params()
    
    print(f"\nParameter structure:")
    print(f"  ✓ intercept: {params['intercept']:.6f}")
    
    print(f"\n  ✓ alpha_s (ability): {len(params['alpha_s'])} users")
    alpha_values = list(params['alpha_s'].values())
    print(f"    - min: {min(alpha_values):.4f}, max: {max(alpha_values):.4f}")
    print(f"    - mean: {np.mean(alpha_values):.4f}, std: {np.std(alpha_values):.4f}")
    
    print(f"\n  ✓ delta_j (difficulty): {len(params['delta_j'])} items")
    delta_values = list(params['delta_j'].values())
    print(f"    - min: {min(delta_values):.4f}, max: {max(delta_values):.4f}")
    print(f"    - mean: {np.mean(delta_values):.4f}, std: {np.std(delta_values):.4f}")
    
    print(f"\n  ✓ beta_k (skill effect): {len(params['beta_k'])} skills")
    beta_values = list(params['beta_k'].values())
    print(f"    - min: {min(beta_values):.4f}, max: {max(beta_values):.4f}")
    print(f"    - mean: {np.mean(beta_values):.4f}, std: {np.std(beta_values):.4f}")
    
    
    
    return params


def test_visualization_roc(plot: Plot.PlotOUTILS, results: dict):
    """Test ROC curve visualization"""
    print(f"\n{'='*70}")
    print("VISUALIZING ROC CURVE")
    print(f"{'='*70}")
    
    try:
        plot.PlotROC(
            TPR=results.get("TPR"),
            FPR=results.get("FPR"),
            AUC=results.get("AUC")
        )
        print("✓ ROC curve plotted successfully")
    except Exception as e:
        print(f"✗ Error plotting ROC: {e}")


def test_visualization_das3h(params: dict, df: pd.DataFrame):
    """Test DAS3H visualization functions"""
    print(f"\n{'='*70}")
    print("TESTING DAS3H VISUALIZATIONS")
    print(f"{'='*70}")
    
    try:
        visualizer = Vis.DAS3HVisualizer(params=params, df=df)
        print("✓ DAS3HVisualizer initialized")
        
        # Test plot_p_vs_ability
        try:
            visualizer.plot_p_vs_ability(
                fixed_item=42,
                user_selection="quantiles",
                n_users=5
            )
            print("✓ plot_p_vs_ability executed successfully")
        except Exception as e:
            print(f"✗ Error in plot_p_vs_ability: {e}")
            
    except Exception as e:
        print(f"✗ Error initializing DAS3HVisualizer: {e}")


def test_item_analysis(df: pd.DataFrame, params: dict, plot: Plot.PlotOUTILS):
    """Analysis of items by difficulty and interactions"""
    print(f"\n{'='*70}")
    print("ITEM ANALYSIS")
    print(f"{'='*70}")
    
    item_counts = df["item_id"].value_counts()
    top_items = item_counts.head(5).index.tolist()
    
    print(f"\nTop 5 most interacted items:")
    for i, item_id in enumerate(top_items, 1):
        n_interactions = item_counts[item_id]
        delta = params["delta_j"].get(item_id, 0.0)
        print(f"  {i}. Item {item_id}: {n_interactions:4d} interactions, difficulty={delta:.3f}")


def ComputeMeanTheta(params:dict=None):
    dicMeanThetaW= {}
    dicMeanThetaA= {}
    Window=[3600, 86400, 604800, 2592000, float("inf")]
    ThetaWins=params["theta_wins"]
    ThetaAttempts=params["theta_attempts"]
    for k in range(len(Window)):
        thetawinsK=[ThetaWins.get(kc)[k] for kc in ThetaWins.keys()]
        thetaattemptsK=[ThetaAttempts.get(kc)[k] for kc in ThetaAttempts.keys()]
        meanThetaW=np.mean(thetawinsK)
        meanThetaA=np.mean(thetaattemptsK)
        dicMeanThetaW[k]=meanThetaW
        dicMeanThetaA[k]=meanThetaA
    return dicMeanThetaW,dicMeanThetaA


def PlotMeanTheta(dicMeanThetaW:dict=None,dicMeanThetaA:dict=None):
    import matplotlib.pyplot as plt
    x = list(dicMeanThetaW.keys())
    y_wins = list(dicMeanThetaW.values())
    y_attempts = list(dicMeanThetaA.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_wins, marker='o', label='Mean Theta Wins')
    plt.plot(x, y_attempts, marker='o', label='Mean Theta Attempts')
    #plt.xscale('log')
    plt.xlabel('Time Window (seconds)')
    plt.ylabel('Mean Theta Value')
    plt.title('Mean Theta Wins and Attempts by Time Window')
    plt.xticks(x, ['1h', '1d', '1w', '30d', 'inf'])
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_wins_distribution(df, kc_list):
    """
    Analyse combien d'élèves ont des succès dans chaque fenêtre.
    """
    from collections import defaultdict
    TW_SECONDS = [3600, 86400, 604800, 2592000, float("inf")]
    TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
    
    counts_by_tw = {tw: [] for tw in TW_LABELS}
    
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].sort_values("timestamp")
        t_current = df_user["timestamp"].max()
        
        wins_timestamps = defaultdict(list)
        
        for _, row in df_user.iterrows():
            if row["correct"] == 1:
                kcs_row = str(row["KC"]).split("~~")
                for kc in kcs_row:
                    if kc in kc_list:
                        wins_timestamps[kc].append(row["timestamp"])
        
        # Compter par fenêtre
        for kc in kc_list:
            if kc not in wins_timestamps:
                continue
            
            timestamps = wins_timestamps[kc]
            for i, (tw_sec, tw_label) in enumerate(zip(TW_SECONDS, TW_LABELS)):
                if tw_sec == float("inf"):
                    n = len(timestamps)
                else:
                    n = sum(1 for t in timestamps if (t_current - t) <= tw_sec)
                
                counts_by_tw[tw_label].append(n)
    
    # Afficher
    print(f"\n{'='*70}")
    print(f"  Distribution des succès par fenêtre temporelle")
    print(f"{'='*70}\n")
    print(f"{'Fenêtre':<10} {'Total':<12} {'Moyenne':<12} {'Médiane':<12} {'% avec n>0':<15}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*15}")
    
    for tw_label in TW_LABELS:
        counts = counts_by_tw[tw_label]
        sum_=np.sum(counts)
        avg = np.mean(counts)
        med = np.median(counts)
        pct_nonzero = sum(1 for c in counts if c > 0) / len(counts) * 100
        
        print(f"{tw_label:<10} {sum_:<12.0f} {avg:>10.2f}   {med:>10.0f}   {pct_nonzero:>13.1f}%")
    
    print(f"{'='*70}\n")

def analyse_eleve(df, X, user, params, model):
    print("The user is :", user)
    print("The level of ability of the user is :", params["alpha_s"].get(user))

    #Difficulty 
    df_u = df[df.user_id == user][["item_id", "correct", "KC", "timestamp"]]
    delta = params["delta_j"] 
    df_compare = df_u.copy()
    df_compare["delta_j"] = df_compare["item_id"].map(delta)
    df_compare.sort_values("delta_j")
    df_compare["bien_estime"] = (
    ((df_compare["delta_j"] > 0) & (df_compare["correct"] == 0)) |
    ((df_compare["delta_j"] < 0) & (df_compare["correct"] == 1)))
    nb_bien = df_compare["bien_estime"].sum()
    nb_total = len(df_compare)
    nb_mal = nb_total - nb_bien
    print(f"\n{'='*70}")
    print("ANALYSE DE L'item")
    print(f"{'='*70}\n")
    print("Bien estimés :", nb_bien)
    print("Mal estimés  :", nb_mal)
    print("Taux de bonne estimation :", nb_bien / nb_total)
    plt.figure(figsize=(10,5))

    plt.scatter(df_compare["delta_j"],df_compare["correct"],c=df_compare["bien_estime"].map({True: "green", False: "red"}),alpha=0.7,s=80)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.6)
    plt.xlabel("delta_j (difficulté estimée)")
    plt.ylabel("correct (0 = échec, 1 = réussite)")
    plt.title("Items bien estimés (vert) vs mal estimés (rouge)")
    plt.show()

    #Kc list 

    df_u["KC_list"] = df_u["KC"].astype(str).str.split("~~")
    df_u = df_u.drop(columns=["KC"])
    df_exploded = df_u.explode("KC_list").rename(columns={"KC_list": "KC"})
    kc_perf = df_exploded.groupby("KC")["correct"].mean()
    params = model.get_params()
    beta = params["beta_k"]

    df_kc = kc_perf.to_frame("accuracy")
    df_kc["beta_k"] = df_kc.index.map(beta)
    df_kc["bien_estime"] = (
        ((df_kc["beta_k"] > 0) & (df_kc["accuracy"] >= 0.5)) |
        ((df_kc["beta_k"] < 0) & (df_kc["accuracy"] < 0.5))
    )
    nb_bien = df_kc["bien_estime"].sum()
    nb_total = len(df_kc)
    nb_mal = nb_total - nb_bien
    print(f"\n{'='*70}")
    print("ANALYSE DES COMPETENCES (KC)")
    print(f"{'='*70}\n")
    print("Bien estimés :", nb_bien)
    print("Mal estimés  :", nb_mal)
    print("Taux de bonne estimation :", nb_bien / nb_total)

    plt.figure(figsize=(8,5))
    plt.scatter(
        df_kc["beta_k"],
        df_kc["accuracy"],
        c=df_kc["bien_estime"].map({True: "green", False: "red"}),
        s=80
    )

    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("beta_k (facilité estimée du KC)")
    plt.ylabel("Accuracy brute de l'élève")
    plt.title("KC bien estimés (vert) vs mal estimés (rouge)")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DAS3H MODEL TEST SUITE")
    print("="*70)
    
    # 1. Load data
    #df, q_matrix = setup_data(DATA_FOLDER, N_STUDENTS)
    
    # 2. Load student model for metadata
    stdmodel,df,q_matrix = load_student_model(DATA_FOLDER,mininteractions=MIN_INTERACTIONS,n_students=N_STUDENTS)
    
    # 3. Prepare features
    X, user_ids, item_ids, kc_list = prepare_features(df, q_matrix, stdmodel)
    
    # 4. Train model and test results
    model, results = test_model_training(
        X, user_ids, item_ids, kc_list,
        model_c=MODEL_C,
        n_tw=N_TIME_WINDOWS
    )
    
    # 5. Test model parameters
    params = test_model_parameters(model)
    #analyse_eleve(df, X, df["user_id"].iloc[0], params, model)
    dicMeanThetaW,dicMeanThetaA=ComputeMeanTheta(params)
    PlotMeanTheta(dicMeanThetaW=dicMeanThetaW,dicMeanThetaA=dicMeanThetaA)
    kc_list = list(params["theta_wins"].keys())
    analyze_wins_distribution(df, kc_list)
    print("Done")
    """# 6. Test visualizations
    plot = Plot.PlotOUTILS()
    test_visualization_roc(plot, results)
    test_visualization_das3h(params, df)
    
    # 7. Item analysis
    test_item_analysis(df, params, plot)
    """
   