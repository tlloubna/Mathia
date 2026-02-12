import matplotlib.pyplot as plt 
import seaborn as sns
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
