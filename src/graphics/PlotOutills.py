import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
class PlotOUTILS : 
    def __init__(self):
        pass
    
    def plot_Q_matrix(self,Q=None, KComp=None, max_items=200):
        
        Q_small = Q[:max_items, :]

        plt.figure(figsize=(12, 8))
        sns.heatmap(Q_small, cmap="Greys", cbar=False)
        plt.xlabel("Skills (KC)")
        plt.ylabel("Items")
        plt.title(f"Matrix Q (display limited to {max_items} items)")
        plt.show()

    def PlotScoreDifficulty(self,score_diff=None):
        sns.heatmap(score_diff[['difficulty_score']], cmap="Reds", annot=False)
        plt.title("Heatmap des difficultés des KC")
        plt.show()
    
    def plot_all_forgetting_curves(self, forgettingDict ):
        plt.figure(figsize=(10, 6))

        t = np.linspace(0, 20000, 200)

        for kc, (a,b) in forgettingDict.items():
            P = 1 / (1 + np.exp(-(a - b * t)))
            plt.plot(t/3600, P, alpha=0.3)  # alpha=0.3 pour transparence

        plt.xlabel("Temps écoulé (heures)")
        plt.ylabel("Probabilité de réussite")
        plt.title("Courbes d’oubli pour toutes les compétences")
        plt.grid()
        plt.show()

    def plotAllProbaIRT(self, irt_model):
        
        plt.figure(figsize=(10, 6))

        for kc in irt_model.kc_categories:
            x, y = irt_model.item_curve(kc)
            plt.plot(x, y, alpha=0.4)

        plt.xlabel("Compétence (ability)")
        plt.ylabel("Probabilité de réussite")
        plt.title("Courbes IRT pour toutes les compétences")
        plt.grid(True)
        plt.show()

    def plotpresence( self,presence,vars):
        kcs = list(presence.keys())
        counts = list(presence.values())

        plt.figure(figsize=(14, 6))
        plt.bar(kcs, counts, color="mediumseagreen", edgecolor="black")
        plt.xticks(rotation=90)
        plt.xlabel(f"{vars[0]}")
        plt.ylabel(f"{vars[1]}")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()

    


