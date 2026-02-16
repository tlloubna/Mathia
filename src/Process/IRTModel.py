import src.datamodel.Studentdata as SD
import numpy as np 
import pandas as pd 

class IRTMODEL: 
    def __init__(self,DataStudent:SD.StudentDATA=None):
        self.data=DataStudent.data.copy()
        #Factorisation 
        self.data["student_id"], self.student_categories = pd.factorize(self.data["user_id"])
        self.data["kc_id"], self.kc_categories = pd.factorize(self.data["KC"])

        self.n_students=len(self.student_categories)
        self.n_kc=len(self.kc_categories)
        #Paramètres du modèle
        self.alpha=np.zeros(self.n_students)
        self.delta=np.zeros(self.n_kc)


    def sigmoidFunc(self,alpha,delta):
        return 1/(1+np.exp(delta-alpha))
    
    def fit(self,lr=0.01,epoches=100):

        df=self.data
        count=0
        for epoch in range(epoches):
            for _,row in df.iterrows():
                print("Step",count,"/",epoches*len(df.iterrows))
                s=row["student_id"]
                j=row["kc_id"]
                y=row["correct_first_attempt"]
                pred =self.sigmoidFunc(self.alpha[s],self.delta)
                error =y- pred 
                self.alpha[s]+=lr*error
                self.delta[j]-= lr * error 

            print(f"Epoch {epoch +1} /{epoches} terminée")
    
    def getStdability(self):
        return pd.DataFrame({
            "student": self.student_categories,
            "ability":self.alpha
        })
    
    def getKcDiff(self):
        return pd.DataFrame({
            "KC": self.kc_categories,
            "ability":self.delta
        })

    def predict(self,std,kc):
        s=np.where(self.student_categories==std)[0][0]
        j=np.where(self.kc_categories==kc)[0][0]
        return self.sigmoidFunc(self.alpha[s],self.delta[j])
    

    def item_curve(self, kc_name, ability_range=None):
        if ability_range is None:
            ability_range = np.linspace(-3, 3, 200)

        # trouver l'index du KC
        j = np.where(self.kc_categories == kc_name)[0][0]
        delta_j = self.delta[j]

        # probabilité pour chaque niveau de compétence
        probs = self.sigmoidFunc(ability_range, delta_j)

        return ability_range, probs

