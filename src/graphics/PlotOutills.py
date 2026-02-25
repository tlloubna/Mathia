import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from utils.this_queue import OurQueue
from sklearn.cluster import KMeans
# Palette commune
BLUE   = "#4C9BE8"
PINK   = "#E85C8A"
GREEN  = "#3DBE8A"
ORANGE = "#F5A623"
PURPLE = "#9B59B6"
COLORS = [BLUE, PINK, GREEN, ORANGE, PURPLE, "#E74C3C", "#1ABC9C", "#F39C12"]
DARK_BG = "#1C1C2E"
CARD_BG = "#252540"

"""plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   "#3A3A5C",
    "axes.labelcolor":  "#CCCCDD",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#CCCCDD",
    "grid.color":       "#2E2E4E",
    "grid.linewidth":   0.6,
})"""


class PlotOUTILS:
    def __init__(self):
        pass

   
    def plot_Q_matrix(self, Q=None, max_items=200):
        Q_small = Q[:max_items, :]
        plt.figure(figsize=(12, 8))
        sns.heatmap(Q_small, cmap="Greys", cbar=False)
        plt.xlabel("Skills (KC)")
        plt.ylabel("Items")
        plt.title(f"Matrix Q (display limited to {max_items} items)")
        plt.show()

    def PlotScoreDifficulty(self, score_diff=None):
        sns.heatmap(score_diff[['difficulty_score']], cmap="Reds", annot=False)
        plt.title("Heatmap des difficultés des KC")
        plt.show()

    def plot_all_forgetting_curves(self, forgettingDict):
        plt.figure(figsize=(10, 6))
        t = np.linspace(0, 40, 2)
        for kc, (a, b) in forgettingDict.items():
            P = 1 / (1 + np.exp(-(a - b * t)))
            plt.plot(t, P, alpha=0.3)
        plt.xlabel("Days")
        plt.ylabel("Probabilité de réussite")
        plt.title("Courbes d'oubli pour toutes les compétences")
        plt.grid()
        plt.show()

    def plotpresence(self, presence, vars):
        kcs    = list(presence.keys())
        counts = list(presence.values())
        plt.figure(figsize=(14, 6))
        plt.bar(kcs, counts, color="mediumseagreen", edgecolor="black")
        plt.xticks(rotation=90)
        plt.xlabel(f"{vars[0]}")
        plt.ylabel(f"{vars[1]}")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def PlotROC(self, TPR, FPR, AUC):
        plt.figure(figsize=(14, 6))
        plt.plot(FPR, TPR, alpha=0.6, color=BLUE, lw=2)
        plt.plot([0, 1], [0, 1], "--", color="#555566", lw=1)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC Curve  (AUC = {AUC:.3f})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def ComputeHteta(self,kc_list,params,nwins,natt,nfails):
         #nwins : le nombre de ressite pour les 4 plages [1 2 3 4 ] {"kc":[1 4 6 0]}
         #natte :{"kc":[1 4 6 0]} :attemps
         #nfails : {kc :5 }
        mem=0
        for kc in kc_list:
            tw_wins=np.array(params["theta_wins"][kc])
            tw_att=np.array(params["theta_attempts"][kc])
            tf=params["theta_fails"].get(kc,0)
            mem+=np.dot(tw_wins,np.log(1+nwins.get(kc)))
            mem+=np.dot(tw_att,np.log(1+natt.get(kc)))
            mem+=tf * nfails.get(kc)
        return mem 

    def ComputeNfailsAttWins(self, user_id, t_current, df, kc_list):
        from collections import defaultdict
        TW_SECONDS = [3600, 86400, 604800, 2592000, float("inf")]
        df_user = df[
            (df["user_id"] == user_id) & 
            (df["timestamp"] < t_current)
        ].copy()
        if df_user.empty:
            return (
                {kc: np.zeros(5) for kc in kc_list},
                {kc: np.zeros(5) for kc in kc_list},
                {kc: 0 for kc in kc_list}
            )
        wins_timestamps  = defaultdict(list)  # {kc: [t1, t2, ...]}
        fails_count      = defaultdict(int)   # {kc: nb_fails}
        attempts_timestamps = defaultdict(list)
        for _, row in df_user.iterrows():
            t       = float(row["timestamp"])
            correct = int(row["correct"])
            kcs_row = str(row["KC"]).split("~~")
            for kc in kcs_row:
                if kc in kc_list:
                    attempts_timestamps[kc].append(t)
                    if correct == 1:
                        wins_timestamps[kc].append(t)
                    else:
                        fails_count[kc] += 1
        nwins  = {}
        natt   = {}  
        nfails = {}
        for kc in kc_list:
            wins_ts = wins_timestamps[kc]
            att_ts  = attempts_timestamps[kc]
            wins_counts = []
            for tw in TW_SECONDS:
                if tw == float("inf"):
                    wins_counts.append(len(wins_ts))
                else:
                    n = sum(1 for t in wins_ts if (t_current - t) <= tw)
                    wins_counts.append(n)
            att_counts = []
            for tw in TW_SECONDS:
                if tw == float("inf"):
                    att_counts.append(len(att_ts))
                else:
                    n = sum(1 for t in att_ts if (t_current - t) <= tw)
                    att_counts.append(n)
            
            nwins[kc]  = np.array(wins_counts, dtype=float)
            natt[kc]   = np.array(att_counts, dtype=float)
            nfails[kc] = fails_count[kc]
        
        return nwins, natt, nfails
        
    def PlotProbVsAbilityAllItems(self, params: dict, df, user_id, 
                                    nb_items: int = 10, t_current=None):
       
        sigmoid = lambda x: 1 / (1 + np.exp(-x))        
        if t_current is None:
            df_user = df[df["user_id"] == user_id]
            if df_user.empty:
                print(f"[ERREUR] Élève {user_id} introuvable dans df")
                return
            t_current = df_user["timestamp"].max()
        clusters = self.CluesterItemSTd(params, n_clusters=100,var="delta_j")
        

        selected_item = clusters[0][:5]
        alphas = np.linspace(-3, 3, 300)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, item_id in enumerate(selected_item):
            item_rows = df[df["item_id"] == item_id]
            if item_rows.empty:
                continue
            kc_str  = item_rows.iloc[0]["KC"]
            kc_list = str(kc_str).split("~~")
            delta_j = params["delta_j"].get(item_id, 0.0)
            beta_total = sum(params["beta_k"].get(kc, 0.0) for kc in kc_list)
            nwins, natt, nfails = self.ComputeNfailsAttWins(
                user_id, t_current, df, kc_list
            )
            hteta = self.ComputeHteta(kc_list, params, nwins, natt, nfails)
            logits = alphas - delta_j + beta_total + hteta + params["intercept"]
            probs  = sigmoid(logits)
            color = COLORS[i % len(COLORS)]
            label = f"Item {item_id} (δ={delta_j:.2f}, mem={hteta:.2f})"
            
            ax.plot(alphas, probs, color=color, lw=2, label=label, alpha=0.85)
            
        
        # Marquer l'ability réelle de l'élève
        real_alpha = params["alpha_s"].get(user_id, 0.0)
        ax.axvline(real_alpha, color=ORANGE, lw=2, linestyle="-", alpha=0.8,
                label=f"Ability élève {user_id} (alpha={real_alpha:.2f})")
        
        ax.set_xlabel("Ability alpha")
        ax.set_ylabel("P(correct)")
        #ax.set_ylim(0, 1)
        ax.set_xlim(-3, 3)
        ax.set_title(
            f"P(correct) en fonction de l'ability \n"
            
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    def CluesterItemSTd(self, params: dict, n_clusters: int = 5,var="alpha_s"):
        sep = params[var]
        ids = list(sep.keys())
        values = np.array(list(sep.values())).reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(values)

        clusters = {i: [] for i in range(n_clusters)}
        for uid, lab in zip(ids, labels):
            clusters[lab].append(uid)

        return clusters
        
    def choose_cluster_with_size(self, clusters, min_size=5, max_size=7):
    
        for cid, var in clusters.items():
            if min_size <= len(var) <= max_size:
                return cid
        return None
    

    def FindStudentsSameMemory(self, params, df, item_id, t_current=None, eps=0.1):
       
        df_item = df[df["item_id"] == item_id]
        if df_item.empty:
            print(f"[ERREUR] item {item_id} introuvable dans df")
            return {}

        if t_current is None:
            t_current = df_item["timestamp"].max()

        kc_str = df_item.iloc[0]["KC"]
        kc_list = str(kc_str).split("~~")
        mem_by_student = {}
        for user_id in df_item["user_id"].unique():

            nwins, natt, nfails = self.ComputeNfailsAttWins(
                user_id, t_current, df, kc_list
            )
            hteta = self.ComputeHteta(kc_list, params, nwins, natt, nfails)
            mem_by_student[user_id] = hteta
        groups = {}
        used = set()

        for u1, m1 in mem_by_student.items():
            if u1 in used:
                continue

            groups[u1] = [u1]
            used.add(u1)

            for u2, m2 in mem_by_student.items():
                if u2 in used:
                    continue

                if abs(m1 - m2) <= eps:
                    groups[u1].append(u2)
                    used.add(u2)

        return groups



    def probabVsdiffAllstudent(self,params:dict, df,item_id,nb_student:int=10,t_current=None):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        if t_current is None:
            df_item = df[df["item_id"] == item_id]
            if df_item.empty:
                print(f"[ERREUR]  {item_id} introuvable dans df")
                return
            t_current = df_item["timestamp"].max() 
        else :
            df_item = df[df["item_id"] == item_id] 
            if df_item.empty:
                print(f"[ERREUR]  {item_id} introuvable dans df")
                return

        clusters = self.CluesterItemSTd(params, n_clusters=5,var="alpha_s")
        cid = self.choose_cluster_with_size(clusters, min_size=5, max_size=7)
        if cid is None:
            print("Aucune classe avec 5 à 7 élèves")
            return

        selected_students = clusters[0][:5]
        Stdsamemem=self.FindStudentsSameMemory(params=params,df=df,item_id=item_id,t_current=t_current,eps=0.3)
        filtered = [ members for leader, members in Stdsamemem.items() if len(members) >= 2 ]
        clean = [[int(x) for x in pair] for pair in filtered]
        if len(clean)<1:
            print("pas de std pour ex x qui partage meme memoire ")
            return
        deltas=np.linspace(-4,4,200)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, user_id in enumerate(clean[0]):
            user_rows = df_item[df_item["user_id"] == user_id]
            if user_rows.empty:
                continue
                
            kc_str  = user_rows.iloc[0]["KC"]
            kc_list = str(kc_str).split("~~")
            alpha_s = params["alpha_s"].get(user_id, 0.0)
            beta_total = sum(params["beta_k"].get(kc, 0.0) for kc in kc_list)
            nwins, natt, nfails = self.ComputeNfailsAttWins(
                user_id, t_current, df, kc_list
            )
            hteta = self.ComputeHteta(kc_list, params, nwins, natt, nfails)
            logits = alpha_s-deltas + beta_total + hteta + params["intercept"]
            probs  = sigmoid(logits)
            color = COLORS[i % len(COLORS)]
            label = f"student {user_id} (α={alpha_s:.2f}, mem={hteta:.2f})"
            
            ax.plot(deltas, probs, color=color, lw=2, label=label, alpha=0.85)

        real_delta= params["delta_j"].get(item_id, 0.0)
        ax.axvline(real_delta, color=ORANGE, lw=2, linestyle="-", alpha=0.8,
                label=f"Difficulty item {user_id} (delta={real_delta:.2f})")
        
        ax.set_xlabel("Difficulty delta")
        ax.set_ylabel("P(correct)")
        #ax.set_ylim(0, 1)
        ax.set_xlim(-3, 3)
        ax.set_title(
            f"P(correct) en fonction de la difficulté\n"
            
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
            








