import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
COLUMN_RENAME = {
    "Anon Student Id": "user_id",
    "Problem Hierarchy": "problem_hierarchy",
    "Problem Name": "item_id",
    "Problem View": "problem_view",
    "Step Name": "step_name",
    "Step Start Time": "start_time",
    "First Transaction Time": "first_transaction_time",
    "Correct Transaction Time": "correct_transaction_time",
    "Step End Time": "end_time",
    "Step Duration (sec)": "duration",
    "Correct Step Duration (sec)": "correct_duration",
    "Error Step Duration (sec)": "error_duration",
    "Correct First Attempt": "correct_first_attempt",
    "Incorrects": "incorrects",
    "Hints": "hints",
    "Corrects": "corrects",
    "KC(Default)": "KC",
    "KC(SubSkills)": "KC",
    "Opportunity(Default)": "opportunity",
    "Opportunity(SubSkills)": "opportunity",
}

class ExploreDATA:
    def __init__(self,file:str="algebra05"):
        self.pathfile=file
        self.data=None
    def loadData(self,Display:bool=False):
        print("Load data...")
        try :
            self.data=pd.read_csv(self.pathfile,sep='\t')
            self.data = self.data.rename(columns=COLUMN_RENAME)
            if Display:
                print("First lines of file",os.path.basename(self.pathfile))
                print(self.data.head(7))
                print("Columuns: ",self.data.columns)
                print("Description : ",self.data.describe())
            return self.data
        except Exception as e:
            print("A problem in loading data")
            return None
        
    def MissingValue(self):
        # Pourcentage de valeurs manquantes
        missing = self.data.isna().mean().sort_values(ascending=False) * 100
        print(missing)
        plt.figure(figsize=(10,6))
        missing.plot(kind="bar")
        plt.title("Percentage of missing values ​​per variable")
        plt.ylabel("% missing")
        plt.show()
        print("Done")
    
    def DistribV(self):
        num_cols = self.data.select_dtypes(include=np.number).columns.tolist()

        n = len(num_cols)
        cols_per_fig = 6 
        rows = 2          

        for i in range(0, n, cols_per_fig):
            subset = num_cols[i:i+cols_per_fig]

            fig, axes = plt.subplots(rows, 3, figsize=(18, 10))
            axes = axes.flatten()

            for ax, col in zip(axes, subset):
                sns.histplot(self.data[col], kde=True, bins=50, ax=ax)
                ax.set_title(f"Distribution de {col}")

            for j in range(len(subset), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()
 

    def MatrixCorr(self):
        num_df = self.data.select_dtypes(include=np.number)
        corr = num_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)

        plt.title("correlation matrix")
        plt.show()



        

            