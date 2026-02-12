
import src.graphics.Exploredata as EXD
import pandas as pd
import numpy as np
import copy
class StudentDATA:
    def __init__(self, file:str="algebra05/data.txt"):
        self.pathfile = file
        self.data = None
        self.users = None
        self.items = None
        self.KComp = None
        self.Q = None
        self.exp1 = EXD.ExploreDATA(file=self.pathfile)

    def loadData(self, Display:bool=False):
        df = self.exp1.loadData(Display=False)

        # Créer item_id = item_name + step_name 
        if "step_name" in df.columns:
            df["item_id"] = df["item_id"].astype(str) + ":" + df["step_name"].astype(str)

        df["item_id"], _ = pd.factorize(df["item_id"])
        df["user_id"], _ = pd.factorize(df["user_id"])

        # Nettoyer les KC
        df = df.dropna(subset=["KC"])
        df["KC"] = df["KC"].astype(str)
        self.data = df
        # Extraire les listes
        self.users = df["user_id"].unique().tolist()
        self.items = df["item_id"].unique().tolist()
        self.KComp = df["KC"].unique().tolist()
        if Display:
            print(f"N° of students : {len(self.users)}")
            print(f"N° of items    : {len(self.items)}")
            print(f"N° of KC       : {len(self.KComp)}")

        return df

    def build_Q_matrix(self):

        df = self.data.copy()
        df["KC"] = df["KC"].str.split("~~")
        df["item_id"], _ = pd.factorize(df["item_id"])
        all_kc = sorted({kc for sublist in df["KC"] for kc in sublist})

        kc_to_idx = {kc: i for i, kc in enumerate(all_kc)}
        n_items = df["item_id"].nunique()
        n_kc = len(all_kc)
        Q = np.zeros((n_items, n_kc), dtype=int)
        count=0
        for _, row in df.iterrows():
            item = int(row["item_id"])
            for kc in row["KC"]:
                count+=1
                Q[item, kc_to_idx[kc]] = 1
            print("Step",count,"/",n_kc*n_items)

        self.Q = Q
        self.KComp = all_kc

        return Q
