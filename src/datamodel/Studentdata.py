
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

    def loadData(self, Display:bool=False, min_intercation:int=20, n_students:int=50):

        # Charger les données brutes
        self.data = self.exp1.loadData(Display=False)

        # Sélectionner un sous-ensemble d'élèves AVANT factorisation
        unique_users = self.data["user_id"].unique()
        selected_users = np.random.choice(unique_users, size=n_students, replace=False)
        self.data = self.data[self.data["user_id"].isin(selected_users)]

        # Construire item_id = item_name + step_name
        if "step_name" in self.data.columns:
            self.data["item_id"] = self.data["item_id"].astype(str) + ":" + self.data["step_name"].astype(str)

        # Nettoyer KC
        self.data = self.data[~self.data["KC"].isnull()]
        self.data["KC"] = self.data["KC"].astype(str)

        # Construire la liste des KC uniques (sans explode)
        listOfKC = []
        for kc_raw in self.data["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)

        # Dictionnaires KC <-> index
        dict1_kc = {kc: i for i, kc in enumerate(listOfKC)}
        dict2_kc = {i: kc for i, kc in enumerate(listOfKC)}

        # Factoriser user_id et item_id APRÈS filtrage
        self.data["user_id"], _ = pd.factorize(self.data["user_id"])
        self.data["item_id"], _ = pd.factorize(self.data["item_id"])

        # Encoder timestamp
        self.data["timestamp"] = pd.to_datetime(self.data["first_transaction_time"])
        self.data["timestamp"] = (self.data["timestamp"] - self.data["timestamp"].min()).dt.total_seconds().astype(int)

        # Renommer correct
        self.data.rename(columns={"correct_first_attempt": "correct"}, inplace=True)

        # Construire Q-matrix
        n_items = self.data["item_id"].nunique()
        n_kc = len(listOfKC)
        Q_mat = np.zeros((n_items, n_kc))

        for _, row in self.data.iterrows():
            item = row["item_id"]
            for kc in row["KC"].split("~~"):
                Q_mat[item, dict1_kc[kc]] = 1

        # Filtrer les élèves avec assez d'interactions
        self.data["inter_id"] = self.data.index
        self.data = self.data.groupby("user_id").filter(lambda x: len(x) >= min_intercation)
        self.data=self.data[['user_id','item_id',"KC","timestamp","correct","inter_id"]]
        self.KComp=listOfKC
        if Display:
            print("Shape Data", self.data.shape)
            print("N° of students :", self.data["user_id"].nunique())
            print("N° of items    :", self.data["item_id"].nunique())
            print("N° of KC       :", len(listOfKC))
            print(self.data.head())

        return self.data, Q_mat



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
        self.data = self.data.dropna(subset=["KC"])
        self.data["KC"] = self.data["KC"].astype(str)
        
        self.data["KC"] = (self.data["KC"].str.split(r"\s*(?:--|~~|-|/|\||;|,)\s*"))
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
    
    