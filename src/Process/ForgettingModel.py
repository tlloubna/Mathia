import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import src.datamodel.Studentdata as STD
import statsmodels.api as sm

class ForgettingMODEL:
    def __init__(self,DataStudent: STD.StudentDATA=None):
        self.stdata=DataStudent.data.copy()
        self.forgettingDict:dict= {}
    
    def PrepareSQ(self):
        df = self.stdata.copy()
        df["KC"] = df["KC"].astype(str).str.split("~~")
        df = df.explode("KC")
        df["start_time"] = pd.to_datetime(df["start_time"])
        df = df.sort_values(["user_id", "KC", "start_time"]) #on regroupe par user_id, puis par competence puis on trie avec le temps 
        #Pour chaque élève, chaque compétence , le temps écoulé depuis la dernire fois qu'il fait 
        df["delta_t"] = df.groupby(["user_id", "KC"])["start_time"].diff().dt.total_seconds()
        df = df.dropna(subset=["delta_t"])

        self.df_seq = df
        return df
    
    def EstimateFG(self):
        """
        ACT_R
        P(correct) = sigmoid(a - b * delta_t)
        """
        df = self.PrepareSQ()
        count=0
        for kc in df["KC"].unique():
            print("Step",count,"/",len(df["KC"].unique()))
            count+=1
            sub = df[df["KC"] == kc]

            if len(sub) < 20:
                continue  

            X = sm.add_constant(sub["delta_t"]) # X= [1 delta1, 1 delta2 ......]
            y = sub["correct_first_attempt"]

            try:
                #on estime le modèle P(correct ) =sigmoid (a + beta*delta_t)
                model = sm.Logit(y, X).fit(disp=False) 
                b = -model.params["delta_t"]  # la vitesse d'oubli 
                a=model.params["const"]
                self.forgettingDict[kc] = (a,b) #niveau initial de maîtrise
            except:
                continue

        return self.forgettingDict



