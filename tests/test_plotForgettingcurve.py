import os
import sys

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import src.datamodel.Historydata as HIS
import src.datamodel.Studentdata as SD

DATA_FOLDER = os.path.join("data", "simulated")

H   = 3600
D   = 3600 * 24
W   = 3600 * 24 * 7
M   = 3600 * 24 * 30
T_M = 3600 * 24 * 30 * 3
S_M = 3600 * 24 * 30 * 6
O_Y = 3600 * 24 * 30 * 12
INF = float("inf")

window_configs = {
    "das3h_original":        [H, D, W, M, INF],
"w_m_3m_6m_1an_inf":     [W, M, T_M, S_M, O_Y, INF],
    "m_3m_6m_1an_inf":       [M, T_M, S_M, O_Y, INF],
           # --- long terme seul, pour isoler l'effet ---
           "3m_6m_1an_inf":         [T_M, S_M, O_Y, INF],
           "6m_1an_inf":            [S_M, O_Y, INF],}


def charger_modele(nom):
    """Recharge le triplet (modele, metadata, mappings) d'une config."""
    bundle = joblib.load(os.path.join(DATA_FOLDER, f"das3h_model_C1_{nom}.pkl"))
    meta = np.load(os.path.join(DATA_FOLDER, f"history_metadata_{nom}.npz"),
                   allow_pickle=True)
    return {
        "model": bundle["model"],
        "user_ids": list(meta["user_ids"]),
        "item_ids": list(meta["item_ids"]),
        "kc_list": list(meta["kc_list"]),
    }


def df_eleve(df, user_id):
    sub = df[df["user_id"] == user_id].sort_values("timestamp").reset_index(drop=True)
    return sub

def proba_une_ligne(model, df_hist, Q_mat, windows, meta, ts_predict, item_id):
    kc_str = "~~".join(str(meta["kc_list"][int(k)]) for k in np.nonzero(Q_mat[item_id])[0])
    cible = pd.DataFrame([{
        "user_id": df_hist["user_id"].iloc[0] if len(df_hist) else -1,
        "item_id": int(item_id),
        "timestamp": int(ts_predict),
        "correct": 0,                       # valeur cible, ignoree en features
        "inter_id": len(df_hist),
        "KC": kc_str,
    }])
    df_aug = pd.concat([df_hist, cible], ignore_index=True)
    for c in ["user_id", "item_id", "timestamp", "correct", "inter_id"]:
        df_aug[c] = df_aug[c].astype("int64")
    df_aug["KC"] = df_aug["KC"].astype(str)
    his = HIS.HistoryDATA(TimeWindow=windows)
    X, _, _, _ = his.ComputeHistoryFeaturesTWKC(
        Q_mat=Q_mat, df=df_aug,
        vocab_users=meta["user_ids"], vocab_items=meta["item_ids"])
    """drop = sorted({0, 1, 2, 3, 4})
    cols = [c for c in range(X.shape[1]) if c not in drop]
    X = X[:, cols]"""
    cols = list(range(X.shape[1]))

    cols.remove(3)
    X_no_col3 = X[:, cols]
    coefs = model.model.named_steps["lr"].coef_.ravel()
    print('les coefs sont', coefs)
    return float(model.model.predict_proba(X_no_col3)[-1, 1])

DELAIS_EXTRA = [O_Y * 2] 
def probas_par_window(df, Q_mat, user_id, item_id, horizons_fictifs=True):
    resultat = {}
    df_u = df_eleve(df, user_id)
    if df_u.empty:
        print(f"eleve {user_id} : aucune interaction")
        return resultat
    anchor = int(df_u["timestamp"].max())    
    for nom, windows in window_configs.items():
        meta = charger_modele(nom)
        model = meta["model"]
        probas = []

        # instants où l'élève a fait l'item cible
        idx_item = df_u.index[df_u["item_id"] == item_id].tolist()

        for i in idx_item:
            df_hist = df_u.iloc[:i]                    # passé complet jusqu'à cet instant
            ts = int(df_u["timestamp"].iloc[i])         # timestamp réel de cette interaction
            probas.append(proba_une_ligne(model, df_hist, Q_mat, windows, meta, ts, item_id))

        if horizons_fictifs:
            delais = [w for w in windows if w != INF] + DELAIS_EXTRA
            for dt in delais:
                ts = anchor + int(dt)
                probas.append(
                    proba_une_ligne(model, df_u, Q_mat, windows, meta, ts, item_id))

        resultat[nom] = probas
        print(f"[{nom}] {len(idx_item)} reelles (item {item_id}) + "
              f"{len(probas) - len(idx_item)} fictives = {len(probas)} probas")
    return resultat
import matplotlib.pyplot as plt
LABELS_DELAIS = {H: "H", D: "D", W: "W", M: "M", T_M: "3M", S_M: "6M",
                 O_Y: "Y", O_Y * 2: "2Y"}  
def plot_forgetting(resultat, window_configs, delais_extra=()):
    fig, ax = plt.subplots(figsize=(5, 5))

    nom0 = next(iter(resultat))
    # délais fictifs réels de la config + les extra → c'est ce nombre qu'on retranche
    n_delais0 = sum(1 for w in window_configs[nom0] if w != INF) + len(delais_extra)
    n_hist = len(resultat[nom0]) - n_delais0

    # tous les délais rencontrés = ceux des configs + les extra
    tous_delais = sorted(
        {w for ws in window_configs.values() for w in ws if w != INF}
        | set(delais_extra)
    )
    pos_delai = {dt: n_hist + i for i, dt in enumerate(tous_delais)}
    all_ps=[]
    for nom, ps in resultat.items():
        # même ordre que dans probas_par_window : délais config PUIS extra
        delais = [w for w in window_configs[nom] if w != INF] + list(delais_extra)
        n_fict = len(delais)
        n_reel = len(ps) - n_fict

        x_hist = list(range(n_reel))
        x_fict = [pos_delai[dt] for dt in delais]
        ax.plot(x_hist + x_fict, ps, marker="o", label=nom)
        all_ps.extend(ps)
        #ax.plot(x_fict, ps[n_reel:], marker="o", label=nom)
    ax.axvline(n_hist - 0.5, color="#888", ls=":", lw=1)
    ax.axhline(0.5, color="#bbb", ls="--", lw=1)

    xticks = [pos_delai[dt] for dt in tous_delais]
    xlabels = [LABELS_DELAIS.get(dt, str(dt)) for dt in tous_delais]
    ax.set_xticks([(n_hist - 1) / 2] + xticks)
    ax.set_xticklabels(["Historique"] + xlabels)
    #ax.set_xlim(n_hist - 0.5, pos_delai[max(pos_delai)] + 0.5)
    lo, hi = min(all_ps), max(all_ps)
    marge = (hi - lo) * 0.05 or 0.01   # le `or` évite marge=0 si tout est plat
    ax.set_ylim(lo - marge, hi + marge)
    ax.set_xlabel("temps →")
    ax.set_ylabel("Proba predite (DAS3H)")
    
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_moyennes(df, Q_mat, users):
    hist = []                                    # une seule colonne historique
    final = {nom: [] for nom in window_configs}
    for u in users:
        for item_id in df[df["user_id"] == u]["item_id"].unique()[:1]:
            d = probas_par_window(df, Q_mat, u, item_id)
            nom0 = next(iter(d))                 # premier modele, pour l'historique
            n_fict0 = sum(1 for w in window_configs[nom0] if w != INF) + len(DELAIS_EXTRA)
            n_reel = len(d[nom0]) - n_fict0
            if n_reel == 0:
                continue
            hist.append(d[nom0][n_reel - 1])    
            for nom, ps in d.items():
                final[nom].append(ps[-1])       

    # historique en premier, puis les modeles
    labels = ["Historique"] + list(final)
    mu = [np.mean(hist)] + [np.mean(final[n]) for n in final]
    sd = [np.std(hist)]  + [np.std(final[n])  for n in final]

    couleurs = ["gold", "#3994d6", "#d38f54", "#5bc15b", "#d66c6c", "#7c5d9a"]

    plt.figure(figsize=(10, 6))
    for i, (lab, m, s) in enumerate(zip(labels, mu, sd)):
        plt.errorbar(i, m, yerr=s, fmt="o", capsize=6, markersize=10,
                    color=couleurs[i], label=lab)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Proba prédite")
    plt.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":

    #data =pd.read_csv("/home/loubna/Code_Projet_Mathia/Mathia/data/algebra_v2/data.txt",sep="\t")
    df = pd.read_csv(os.path.join(DATA_FOLDER, "preprocessed_data_simulated_1000std.csv"))
    Q = sparse.load_npz(os.path.join(DATA_FOLDER, "q_mat_simulated.npz")).toarray()
    """mappings=joblib.load(os.path.join(DATA_FOLDER, "id_mappings.pkl"))
    user_mapping = mappings["user_mapping"]
    item_mapping = mappings["item_mapping"]"""
    #list_student=["204848","204847","204840","204842","204841"]  #list(user_mapping.keys())[:10]
    #ITEM =list(item_mapping.keys())[0]
    #users = [user_mapping.get(u) for u in data['Anon Student Id'].unique()[2:3]]

    plot_moyennes(df, Q, df["item_id"].unique())
    """for user in df["item_id"].unique()[10:20]:
        USER =user
        ITEM=df[df["user_id"]==user]["item_id"].iloc[0]
        d = probas_par_window(df, Q, USER, ITEM)
        plot_forgetting(d, window_configs, delais_extra=DELAIS_EXTRA)"""

    print("done")