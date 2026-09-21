import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)
import src.datamodel.Studentdata as SD 
import src.datamodel.Historydata as HIS 
import src.Process.DAS3H as DAS3H 
import numpy as np
import pandas as pd

import joblib
from scipy import sparse

NAME_FOLDER="Rawson"
DATA_FOLDER=os.path.join("data",NAME_FOLDER)

def predict_s2_interaction(model, df_hist,Q_mat, meta,ts_predict,item_id,user_id):
    kc_indices = np.nonzero(Q_mat[item_id])[0]
    kc_str = "~~".join(
        str(meta["kc_list"][int(k)])
        for k in kc_indices
    )
    target = pd.DataFrame([{ "user_id": user_id, "item_id": int(item_id), "timestamp": int(ts_predict),"correct": 0,         "inter_id": len(df_hist),"KC": kc_str,}])
    df_aug = pd.concat([df_hist, target],ignore_index=True)
    df_aug["user_id"] = df_aug["user_id"].astype("int64")
    df_aug["item_id"] = df_aug["item_id"].astype("int64")
    df_aug["timestamp"] = df_aug["timestamp"].astype("int64")
    df_aug["correct"] = df_aug["correct"].astype("int64")
    df_aug["inter_id"] = np.arange(len(df_aug))
    df_aug["KC"] = df_aug["KC"].astype(str)
    his = HIS.HistoryDATA()
    X, _, _, _ = his.ComputeHistoryFeaturesTWKC(Q_mat=Q_mat,df=df_aug,vocab_users=meta["user_ids"], vocab_items=meta["item_ids"] )
    cols = list(range(X.shape[1]))
    cols.remove(3)
    X_model = X[:, cols]
    proba = model.model.predict_proba(X_model )[-1, 1]
    return float(proba)

if __name__=="__main__":


    time=3
    if time==0:
        #preprocess
        pathMathiadata = os.path.join(DATA_FOLDER, "..", NAME_FOLDER, "sess1.csv")
        stdmodel :SD.StudentDATA = SD.Mathiadata(pathMathiadata,seed=42)
        df ,Q= stdmodel.loadData(Display=True, min_intercation=1,n_students=99)
        df.to_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_sess1.csv"), index=False)
        sparse.save_npz(os.path.join(DATA_FOLDER, f"q_mat_sess1.npz"), sparse.csr_matrix(Q))
        #features
        his = HIS.HistoryDATA(stdmodel=stdmodel)
        X, user_ids, item_ids, listKC = his.ComputeHistoryFeaturesTWKC(Q_mat=Q, df=df)
        sparse.save_npz(os.path.join(DATA_FOLDER, f"history_features_sess1.npz"), sparse.csr_matrix(X))
        np.savez(os.path.join(DATA_FOLDER, f"history_metadata_sess1.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listKC)
        #training
        model = DAS3H.DAS3HModel(C=0.1)
        results = model.fit(X, user_ids=user_ids, item_ids=item_ids, 
                            kc_list=listKC, perc_init=0.8)
        joblib.dump(
            {"mode": "production", "model": model, "results": results},
            os.path.join(DATA_FOLDER, f"das3h_model_sess1_0.8.pkl")
        )
        print(f"  AUC:  {results['AUC']:.4f}")
        print(f"  NLL:  {results['NLL']:.4f}")
        print(f"  RMSE: {results['RMSE']:.4f}")
        print("done")

    if time == 1:
        p1 = os.path.join(DATA_FOLDER, "..", NAME_FOLDER, "sess1.csv")
        p2 = os.path.join(DATA_FOLDER, "..", NAME_FOLDER, "sess2.csv")
        model = joblib.load(os.path.join(DATA_FOLDER, "das3h_model_sess1.pkl"))["model"]
        meta  = np.load(os.path.join(DATA_FOLDER, "history_metadata_sess1.npz"), allow_pickle=True)
        Q     = sparse.load_npz(os.path.join(DATA_FOLDER, "q_mat_sess1.npz")).toarray()
        stdmodel = SD.Mathiadata(p1, seed=42)
        sess1, _ = stdmodel.loadData(Display=False, min_intercation=1, n_students=99)
        user_map, item_map = stdmodel.user_mapping, stdmodel.item_mapping
        raw1 = pd.read_csv(p1)
        raw1["u"] = raw1["user_id"].map(user_map)
        raw1["i"] = raw1["item_id"].map(item_map)
        cond = (raw1[["u", "i", "Assigned_Criterion", "Learning_Condition_Short_Label"]]
                .dropna(subset=["u", "i"])
                .drop_duplicates(["u", "i"]))
        s2 = pd.read_csv(p2)
        s2["user_id"] = s2["user_id"].map(user_map)
        s2["item_id"] = s2["item_id"].map(item_map)
        s2 = s2.dropna(subset=["user_id", "item_id"]).astype({"user_id": int, "item_id": int})
        s2["timestamp"] = (pd.to_datetime(s2["timestamp"], utc=True, errors="coerce")
                            .astype("int64") // 10**9)
        s2["correct"] = pd.to_numeric(s2["correct"], errors="coerce").fillna(0).astype(int)
        s2["KC"] = s2["KC"].astype(str)
        s2 = s2.sort_values(["user_id", "order_id"])
        s2 = s2.groupby(["user_id", "item_id"], as_index=False).head(1)
        s2 = s2[["user_id", "item_id", "correct", "timestamp", "KC"]] 
        s2 = s2.merge(cond, left_on=["user_id", "item_id"], right_on=["u", "i"])
        sess1 = sess1.sort_values(["user_id", "timestamp"])
        preds = []
        for uid, g in s2.groupby("user_id"):
            hist = sess1[sess1["user_id"] == uid].sort_values("timestamp")
            if hist.empty:
                continue
            for _, r in g.iterrows():
                p = predict_s2_interaction(model=model, df_hist=hist, Q_mat=Q, meta=meta,
                                        ts_predict=int(r["timestamp"]),
                                        item_id=int(r["item_id"]), user_id=int(uid))
                preds.append({"user_id": int(uid), "item_id": int(r["item_id"]),
                            "correct": int(r["correct"]), "prediction": p,
                            "criterion": int(r["Assigned_Criterion"]),
                            "lag": r["Learning_Condition_Short_Label"]})

        df_pred = pd.DataFrame(preds)
        df_pred.to_csv(os.path.join(DATA_FOLDER, "sess2_predictions.csv"), index=False)
        print(df_pred.groupby(["criterion", "lag"])[["correct", "prediction"]].mean())
        print("done")
    if time == 3:
        import matplotlib.pyplot as plt

        df = pd.read_csv(os.path.join(DATA_FOLDER, "sess2_predictions.csv"))
        ordre = ["Lag 7", "Lag 15", "Lag 47"]
        df["lag"] = pd.Categorical(df["lag"], categories=ordre, ordered=True)

        """agg = (df.groupby(["criterion", "lag"], observed=True)
                .agg(actual=("correct", "mean"),
                    predicted=("prediction", "mean"),
                    sd_pred=("prediction", "std"),
                    n=("correct", "size")).reset_index())
        agg["ci_actual"] = 1.96 * np.sqrt(agg["actual"] * (1 - agg["actual"]) / agg["n"])
        agg["ci_pred"]   = 1.96 * agg["sd_pred"] / np.sqrt(agg["n"])"""
        df["pred_bin"] = (df["prediction"] > 0.5).astype(int)

        agg = (df.groupby(["criterion", "lag"], observed=True)
             .agg(actual=("correct", "mean"),
                  predicted=("pred_bin", "mean"),
                  n=("correct", "size")).reset_index())
        agg["ci_actual"] = 1.96 * np.sqrt(agg["actual"]*(1-agg["actual"])/agg["n"])
        agg["ci_pred"]   = 1.96 * np.sqrt(agg["predicted"]*(1-agg["predicted"])/agg["n"])

        crits = sorted(df["criterion"].unique())          # lignes : [1, 3]
        sources = [("Actual Prop Correct", "actual", "ci_actual", "black"),
                ("DAS3H",               "predicted", "ci_pred", "steelblue")]  # colonnes

        fig, axes = plt.subplots(len(crits), 2, figsize=(8, 8),
                         sharex=True, sharey="row", squeeze=False)

        for i, c in enumerate(crits):
            s = agg[agg["criterion"] == c].sort_values("lag")
            x = s["lag"].astype(str)
            for j, (title, ycol, cicol, color) in enumerate(sources):
                ax = axes[i, j]
                ax.errorbar(x, s[ycol], yerr=s[cicol], fmt="o", capsize=4, color=color)
                ax.grid(False)    
                if i == 0:
                    ax.set_title(title)         
                if j == 1:
                    ax.set_ylabel(f"Criterion {c}", rotation=270, labelpad=15)
                    ax.yaxis.set_label_position("right")  
            axes[i, 0].set_ylabel("% Correct")

        plt.tight_layout(); plt.show()
        print(agg[["criterion", "lag", "actual", "predicted", "n"]].to_string(index=False))
