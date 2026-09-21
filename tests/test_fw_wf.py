"""
TEST DÉCISIF — une seule question, une réponse binaire.
DAS3H capte-t-il l'oubli quand les données le contiennent ?

On entraîne DAS3H sur le MÊME dataset simulé, deux fois :
  A) SANS fenêtres  : TimeWindow = [inf]        -> aucune info temporelle
  B) AVEC fenêtres  : TimeWindow = [1h..inf]    -> info d'oubli disponible

Si AUC(B) > AUC(A) de façon nette, les fenêtres apportent le signal d'oubli
=> DAS3H le capte. Sinon, le mécanisme n'aide pas, même en conditions idéales.

À lancer depuis la racine de ton projet (là où 'src' et 'data' sont visibles),
après avoir mis le CSV et la Q-matrice simulés dans data/simulated/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy import sparse
# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)


import src.datamodel.Historydata as HIS
import src.datamodel.Studentdata as SD
import src.Process.DAS3H as DAS3H

DATA_FOLDER = os.path.join("data", "simulated")
CSV ="/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/preprocessed_data_simulated_1000std.csv"
QMAT = "/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/q_mat_1000std.npz"

HOUR = 3600; DAY = 24 * HOUR
BINS = [0, HOUR, DAY, 7 * DAY, 30 * DAY, np.inf]
LABELS = ["<1h", "1h-1j", "1j-7j", "7j-30j", ">30j"]
BIN_CENTERS_DAYS = [0.02, 0.5, 4, 18, 60]   # delai representatif par bin, en jours
HALFLIFE_DAYS = 7.0
WINDOWS = [3600, 86400, 604800, 2592000, float("inf")]
 
 
def retention_reelle(dt_days):
    tau = HALFLIFE_DAYS / np.log(2)
    return np.exp(-dt_days / tau)
 
 
def split_train_predict(df, n_train_rev=2):
    df = df.sort_values(["user_id", "KC", "timestamp"]).copy()
    df["rev_idx"] = df.groupby(["user_id", "KC"]).cumcount()
    df["dt"] = df.groupby(["user_id", "KC"])["timestamp"].diff()
    train = df[df["rev_idx"] < n_train_rev].copy()
    futur = df[df["rev_idx"] >= n_train_rev].copy()
    futur["bin"] = pd.cut(futur["dt"], bins=BINS, labels=LABELS, include_lowest=True)
    return train, futur
 
 
def entrainer_das3h(train_df, Q):
    """Entraine DAS3H sur le train seul (perc_init=1.0 -> tout en train)."""
    his = HIS.HistoryDATA(TimeWindow=WINDOWS)
    X, user_ids, item_ids, kc_list = his.ComputeHistoryFeaturesTWKC(Q_mat=Q, df=train_df)
    model = DAS3H.DAS3HModel(C=1.0)
    model.fit(X, user_ids=user_ids, item_ids=item_ids,
              kc_list=kc_list, n_tw=len(WINDOWS), perc_init=1.0)
    return model, his
 
 
def predire_futur(model, train_df, futur_df, Q):
    """Pour chaque ligne du futur, reconstruit l'historique (train du meme user)
    + la ligne cible, calcule les features, et predit la proba."""
    probas = []
    for _, row in futur_df.iterrows():
        hist = train_df[train_df["user_id"] == row["user_id"]]
        cible = pd.DataFrame([{
            "user_id": int(row["user_id"]), "item_id": int(row["item_id"]),
            "timestamp": int(row["timestamp"]), "correct": 0,
            "inter_id": len(hist), "KC": str(row["KC"]),
        }])
        df_aug = pd.concat([hist, cible], ignore_index=True)
        df_aug["KC"] = df_aug["KC"].astype(str)
        his = HIS.HistoryDATA(TimeWindow=WINDOWS)
        X, _, _, _ = his.ComputeHistoryFeaturesTWKC(
            Q_mat=Q, df=df_aug,
            vocab_users=model.user_ids, vocab_items=model.item_ids)
        cols = list(range(X.shape[1])); cols.remove(3)
        probas.append(float(model.model.predict_proba(X[:, cols])[-1, 1]))
    return np.array(probas)
 
 
def metrics_par_bin(futur, proba, proba_nulle=1.0):
    f = futur.copy(); f["p"] = proba
    rows = []
    for lab in LABELS:
        sub = f[f["bin"] == lab]
        if len(sub) == 0:
            rows.append((lab, np.nan, np.nan, np.nan, np.nan, 0)); continue
        y = sub["correct"].values; p = sub["p"].values
        rows.append((lab, np.mean(p - y), np.sqrt(np.mean((p - y) ** 2)),
                     np.mean(proba_nulle - y), np.sqrt(np.mean((proba_nulle - y) ** 2)),
                     len(sub)))
    return pd.DataFrame(rows, columns=["bin", "biais", "rmse", "biais_nul", "rmse_nul", "n"])
 
 
def plot_biais(m):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    x = range(len(LABELS))
    ax[0].plot(x, m["biais"], "o-", label="DAS3H", lw=2)
    ax[0].plot(x, m["biais_nul"], "s--", color="grey", label="modele nul (predit 1)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(LABELS, rotation=45)
    ax[0].set_ylabel("Biais (proba predite − reel)"); ax[0].set_xlabel("Horizon de delai")
    ax[0].set_title("Biais par horizon\n>0 = surestime la retention (oubli non capte)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(x, m["rmse"], "o-", label="DAS3H", lw=2)
    ax[1].plot(x, m["rmse_nul"], "s--", color="grey", label="modele nul")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(LABELS, rotation=45)
    ax[1].set_ylabel("RMSE"); ax[1].set_xlabel("Horizon de delai")
    ax[1].set_title("RMSE par horizon"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); plt.show()
 
 
def plot_retention(futur, proba):
    """Retention predite (moyenne proba par bin) vs retention reelle connue."""
    f = futur.copy(); f["p"] = proba
    pred = [f[f["bin"] == lab]["p"].mean() for lab in LABELS]
    reel_obs = [f[f["bin"] == lab]["correct"].mean() for lab in LABELS]
    reel_theo = [retention_reelle(d) for d in BIN_CENTERS_DAYS]
    # normaliser la courbe theorique a l'echelle observee (depart identique)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    x = range(len(LABELS))
    ax.plot(x, reel_obs, "o-", color="black", lw=2, label="reussite reelle (data)")
    ax.plot(x, pred, "s-", color="steelblue", lw=2, label="proba predite DAS3H")
    ax.set_xticks(list(x)); ax.set_xticklabels(LABELS, rotation=45)
    ax.set_ylabel("Probabilite de reussite"); ax.set_xlabel("Horizon de delai")
    ax.set_title("Retention predite vs reelle (simule, oubli connu expo 7j)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); plt.show()
 
 
if __name__ == "__main__":
    df = pd.read_csv(CSV); df["KC"] = df["KC"].astype(str)
    Q = sparse.load_npz(QMAT).toarray()
 
    train, futur = split_train_predict(df, n_train_rev=2)
    print(f"Train: {len(train)} | Futur: {len(futur)}")
 
    model, _ = entrainer_das3h(train, Q)
    proba = predire_futur(model, train, futur, Q)
 
    m = metrics_par_bin(futur, proba, proba_nulle=1.0)
    print("\n=== Biais / RMSE par horizon (DAS3H vs nul) ===")
    print(m.to_string(index=False))
 
    plot_biais(m)
    plot_retention(futur, proba)
    print("\nDone.")
 