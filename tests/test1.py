"""
Protocole d'extrapolation temporelle (Schuetze, Yan & Carvalho 2025) applique a DAS3H.

Idee (leur Analyse 2/3) : on N'evalue PAS en retrospectif. On entraine sur les
PREMIERES revisions de chaque (eleve, skill), puis on predit les revisions
SUIVANTES (= horizons de delai croissants), et on compare la prediction a la
realite. Metriques du papier :
  - BIAIS  = moyenne(proba_predite - correct_reel)   -> >0 = surestime (leur resultat)
  - RMSE   = sqrt(moyenne((proba_predite - correct_reel)^2))
  - MODELE NUL = predit toujours 1 (plancher, comme dans le papier)
On agrege ces metriques PAR HORIZON DE DELAI (bin temporel depuis la derniere revision),
ce qui donne la "courbe" : le biais augmente-t-il quand le delai grandit ?
Sur le simule, on connait la retention reelle (expo, demi-vie 7j) -> on trace
aussi retention predite vs reelle.

=> Version AUTONOME de demonstration : ici un "modele" logistique jouet remplace
   DAS3H pour VALIDER LA LOGIQUE du protocole. La version branchee sur ton
   HistoryDATA + DAS3HModel est fournie ensuite (memes fonctions, autre fit/predict).
"""
import numpy as np
import pandas as pd

HOUR=3600; DAY=24*HOUR
BINS=[0,HOUR,DAY,7*DAY,30*DAY,np.inf]
LABELS=["<1h","1h-1j","1j-7j","7j-30j",">30j"]
HALFLIFE_DAYS=7.0

def retention_reelle(dt): 
    tau=HALFLIFE_DAYS*DAY/np.log(2)
    return np.exp(-dt/tau)

def split_train_predict(df, n_train_rev=2):
    """Pour chaque (user,KC) : les n_train_rev premieres revisions -> TRAIN,
    le reste -> a PREDIRE (le futur). Retourne deux DataFrames."""
    df=df.sort_values(["user_id","KC","timestamp"])
    df["rev_idx"]=df.groupby(["user_id","KC"]).cumcount()
    train=df[df["rev_idx"]<n_train_rev].copy()
    futur=df[df["rev_idx"]>=n_train_rev].copy()
    # delai depuis la derniere revision du meme (user,KC), pour binner le futur
    df["dt"]=df.groupby(["user_id","KC"])["timestamp"].diff()
    futur=futur.merge(df[["user_id","item_id","timestamp","dt"]],
                      on=["user_id","item_id","timestamp"],how="left")
    futur["bin"]=pd.cut(futur["dt"],bins=BINS,labels=LABELS,include_lowest=True)
    return train,futur

def metrics_par_bin(futur, proba_predite, proba_nulle=1.0):
    """Biais et RMSE par horizon de delai, modele vs nul."""
    f=futur.copy()
    f["p"]=proba_predite
    rows=[]
    for lab in LABELS:
        sub=f[f["bin"]==lab]
        if len(sub)==0: 
            rows.append((lab,np.nan,np.nan,np.nan,np.nan,0)); continue
        y=sub["correct"].values; p=sub["p"].values
        biais=np.mean(p-y); rmse=np.sqrt(np.mean((p-y)**2))
        biais_nul=np.mean(proba_nulle-y); rmse_nul=np.sqrt(np.mean((proba_nulle-y)**2))
        rows.append((lab,biais,rmse,biais_nul,rmse_nul,len(sub)))
    return pd.DataFrame(rows,columns=["bin","biais","rmse","biais_nul","rmse_nul","n"])

# ---- DEMO de la logique avec un modele jouet ----
if __name__=="__main__":
    df=pd.read_csv("/home/loubna/Code_Projet_Mathia/Mathia/data/simulated/preprocessed_data_simulated_1000std.csv")
    df["KC"]=df["KC"].astype(str)
    train,futur=split_train_predict(df,n_train_rev=2)
    print(f"Train : {len(train)} lignes | Futur a predire : {len(futur)} lignes")

    # MODELE JOUET : proba = taux de reussite moyen observe dans le train pour ce (user,KC)
    # (ne connait PAS le temps -> cense surestimer aux longs delais, comme le papier)
    taux=train.groupby(["user_id","KC"])["correct"].mean().to_dict()
    glob=train["correct"].mean()
    p_jouet=futur.apply(lambda r: taux.get((r["user_id"],r["KC"]),glob),axis=1).values

    m=metrics_par_bin(futur,p_jouet,proba_nulle=1.0)
    print("\n=== Biais et RMSE par horizon (modele jouet SANS temps vs nul) ===")
    print(m.to_string(index=False))
    print("\nLecture : si 'biais' augmente avec le delai = le modele surestime")
    print("de plus en plus la retention (= il ne capte pas l'oubli).")