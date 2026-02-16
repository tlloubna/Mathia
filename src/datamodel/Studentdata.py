
import src.graphics.Exploredata as EXD
import pandas as pd
import numpy as np
import copy
import re
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
        self.data = self.exp1.loadData(Display=False)

        # Créer item_id = item_name + step_name 
        if "step_name" in self.data.columns:
            self.data["item_id"] = self.data["item_id"].astype(str) + ":" + self.data["step_name"].astype(str)

        self.data["item_id"], _ = pd.factorize(self.data["item_id"])
        self.data["user_id"], _ = pd.factorize(self.data["user_id"])

        # Nettoyer les KC
        self.data = self.data.dropna(subset=["KC"])
        self.data["KC"] = self.data["KC"].astype(str)
        
        self.data["KC"] = (self.data["KC"].str.split(r"\s*(?:--|~~|-|/|\||;|,)\s*"))
        self.data = self.data.explode("KC")
        # Extraire les listes
        self.users = self.data["user_id"].unique().tolist()
        self.items = self.data["item_id"].unique().tolist()
        self.KComp = self.data["KC"].unique().tolist()
        if Display:
            print("Shape Data",self.data.shape)
            print(f"N° of students : {len(self.users)}")
            print(f"N° of items    : {len(self.items)}")
            print(f"N° of KC       : {len(self.KComp)}")

        return self.data
    
    def dicSIandIK(self):
        df = self.data.copy()
        

        dicSI = (
            df.groupby("user_id")["item_id"]
            .apply(list)
            .to_dict()
        )

        dicKI = (
            df.groupby("item_id")["KC"]
            .apply(list)
            .to_dict()
        )

        return dicSI, dicKI


    def CleanData(self, Nbofseenitem=30,NbofseenKc=300):

        
        #Garder que les étudiants dont count<<Nbofseen 
        dicSI = (
            self.data.groupby("item_id")["user_id"]
            .apply(list)
            .to_dict()
        )
        dicSI = {item: len(std) for item, std in dicSI.items()}
        item_to_remove = [item for item, count in dicSI.items() if count < Nbofseenitem]
        self.data = self.data[~self.data["item_id"].isin(item_to_remove)]
        self.data = self.data.reset_index(drop=True)
        #garder que les compétences dans count<NbofseenKc
        dicIK = (
            self.data.groupby("KC")["item_id"]
            .apply(list)
            .to_dict()
        )
        dicIK = {KC: len(item) for KC,item in dicIK.items()}
        KC_to_remove = [KC for KC, count in dicIK.items() if count < NbofseenKc]
        self.data = self.data[~self.data["KC"].isin(KC_to_remove)]
        self.data = self.data.reset_index(drop=True)

        self.users = self.data["user_id"].unique().tolist()
        self.items = self.data["item_id"].unique().tolist()
        self.KComp = self.data["KC"].unique().tolist()
        print("***********After Cleaning**********************")
        print(f"N° of students : {len(self.users)}")
        print(f"N° of items    : {len(self.items)}")
        print(f"N° of KC       : {len(self.KComp)}")
        return self.data








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
