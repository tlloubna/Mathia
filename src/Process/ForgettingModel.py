import numpy as np
import pandas as pd
import src.datamodel.Studentdata as STD
import statsmodels.api as sm

class ForgettingMODEL:
    def __init__(self,DataStudent: STD.StudentDATA=None):
        self.stdata=DataStudent.data.copy()
        self.forgettingDict:dict= {}
    
    def ComputeDeltaPerKC(self):
        self.stdata.sort_values(["user_id", "KC", "timestamp"], inplace=True)
        self.stdata["delta_t"] = (
            self.stdata["timestamp"] -
            self.stdata.groupby(["user_id", "KC"])["timestamp"].shift(1)
        ).fillna(0)

        self.stdata["delta_days"] = self.stdata["delta_t"] / 86400.0
        

    
    def EstimateFGKC(self):
        
        #ACT_R
        #P(correct) = sigmoid(a - b * delta_t)
    
        self.ComputeDeltaPerKC()
        df = self.stdata
        count=0
        for kc in df["KC"].unique():
            print("Step",count,"/",len(df["KC"].unique()))
            count+=1
            sub = df[df["KC"] == kc]

            if len(sub) < 20:
                continue  
            #DeltaT: temps écoulé depuis la dernière fois que l’élève a pratiqué ce KC
            X = sm.add_constant(sub["delta_t"]) # X= [1 delta1, 1 delta2 ......]
            y = sub["correct"]

            try:
                #on estime le modèle P(correct ) =sigmoid (a + beta*delta_t)
                model = sm.Logit(y, X).fit(disp=False) 
                b = -model.params["delta_t"]  # la vitesse d'oubli 
                a=model.params["const"]
                self.forgettingDict[kc] = (a,b) #niveau initial de maîtrise
            except:
                continue

        return self.forgettingDict

    def EstimateFGUser(self):
        """
        ACT-R forgetting model per student:
        P(correct) = sigmoid(a_u - b_u * delta_t)
        """
        self.ComputeDeltaPerKC()
        df = self.stdata

        forgetting = {}

        for u in df["user_id"].unique():
            sub = df[df["user_id"] == u]

            if len(sub) < 20:
                continue

            X = sm.add_constant(sub["delta_days"])
            y = sub["correct"]

            try:
                model = sm.Logit(y, X).fit(disp=False)
                a = model.params["const"]
                b = -model.params["delta_days"]
                forgetting[u] = (a, b)
            except:
                continue

        self.forgettingDict = forgetting
        return forgetting



