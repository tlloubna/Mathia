
import src.graphics.Exploredata as EXD
import pandas as pd
import numpy as np
import copy
import re
class StudentDATA:
    def __init__(self, file:str="algebra05/data.txt",seed:int=42):
        self.pathfile = file #if it csv 
        self.data = None
        self.users = None
        self.items = None
        self.KComp = None
        self.Q = None
        self.exp1 = EXD.ExploreDATA(file=self.pathfile)
        self.seed = seed

    def loadData(self, Display:bool=False, min_intercation:int=20, n_students:int=50):

        # Charger les données brutes
        self.data = self.exp1.loadData(Display=False)

        # Sélectionner un sous-ensemble d'élèves AVANT factorisation
        unique_users = self.data["user_id"].unique()
        rng = np.random.default_rng(self.seed)  # générateur NumPy avec seed
        selected_users = rng.choice(unique_users, size=n_students, replace=False)
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
        self.data["user_id"], user_uniques = pd.factorize(self.data["user_id"])
        self.data["item_id"], item_uniques = pd.factorize(self.data["item_id"])
        self.user_mapping = {orig: fact for fact, orig in enumerate(user_uniques)}
        self.item_mapping = {orig: fact for fact, orig in enumerate(item_uniques)}

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
        self.Q=Q_mat
        self.users = self.data["user_id"].unique().tolist()
        self.items = self.data["item_id"].unique().tolist()
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
    
    def get_student_kcs(self, data, student_id):
        KC_student = data[data["user_id"] == student_id]["KC"]
        kcs = []
        for kc_raw in KC_student.unique():
            for elt in kc_raw.split("~~"):
                kcs.append(elt)
        return kcs

class Mathiadata(StudentDATA):
    def __init__(self, file:str="data/Mathiadata/data.csv",seed:int=42):
        super().__init__(file,seed)
    def loadData(self, Display:bool=False, min_intercation:int=20, n_students:int=50):
        df = pd.read_csv(self.pathfile)

        df = df.rename(columns={
            "student_id": "user_id",
            "kc_names":     "KC",
        })
        #le nombre de sec écoulées depuis le 1ier janvier 1970 à 00:00:00 UTC
        #1970 est la date de la naisance de 1970
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9 #on convertit en ns puis en sec 

        df = df[df["KC"].notna()]
        df["KC"] = df["KC"].astype(str)

        unique_users = df["user_id"].unique()
        rng = np.random.default_rng(self.seed)
        if n_students < len(unique_users):
            selected = rng.choice(unique_users, size=n_students, replace=False)
            df = df[df["user_id"].isin(selected)]

        df["inter_id"] = df.index
        df = df.groupby("user_id").filter(lambda x: len(x) >= min_intercation)
        df = df.reset_index(drop=True)
        
        df["user_id"], user_uniques = pd.factorize(df["user_id"])
        df["item_id"], item_uniques = pd.factorize(df["item_id"])
        self.user_mapping = {orig: fact for fact, orig in enumerate(user_uniques)}
        self.item_mapping = {orig: fact for fact, orig in enumerate(item_uniques)}
        df["inter_id"] = df.index

        all_kcs = []
        for kc_raw in df["KC"].unique():
            for kc in str(kc_raw).split("~~"):
                all_kcs.append(kc)
        all_kcs = np.unique(all_kcs)
        kc_to_idx = {kc: i for i, kc in enumerate(all_kcs)}

        
        n_kc    = len(all_kcs)
        Q_mat = np.zeros((df["item_id"].nunique(), len(all_kcs)), dtype=int)
        item_skill = np.array(df[["item_id", "KC"]].drop_duplicates())  

        for i in range(len(item_skill)):
            for kc in str(item_skill[i, 1]).split("~~"):
                if kc in kc_to_idx:
                    Q_mat[int(item_skill[i, 0]), kc_to_idx[kc]] = 1

        df = df[["user_id", "item_id", "KC", "timestamp", "correct", "inter_id"]]
        self.data   = df
        self.Q      = Q_mat
        self.KComp  = all_kcs.tolist()
        self.users  = df["user_id"].unique().tolist()
        self.items  = df["item_id"].unique().tolist()

        if Display:
            print(f"Shape Data     : {df.shape}")
            print(f"N° of students : {df['user_id'].nunique()}")
            print(f"N° of items    : {df['item_id'].nunique()}")
            print(f"N° of KC       : {n_kc}")
            print(df.head())

        return df, Q_mat