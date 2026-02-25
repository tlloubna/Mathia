import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

# Palette
BLUE   = "#4C9BE8"
PINK   = "#E85C8A"
GREEN  = "#3DBE8A"
ORANGE = "#F5A623"
PURPLE = "#9B59B6"
COLORS = [BLUE, PINK, GREEN, ORANGE, PURPLE, "#E74C3C", "#1ABC9C", "#F39C12"]


class DAS3HVisualizer:
    
    
    def __init__(self, params, df):
        self.params = params
        self.df = df
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.TW_SECONDS = [3600, 86400, 604800, 2592000, float("inf")]
    
    
    def compute_memory(self, user_id, t_current, kc_list):
        """Calcule la contribution mémoire pour un élève sur des KC donnés."""
        nwins, natt, nfails = self._compute_history(user_id, t_current, kc_list)
        
        mem = 0.0
        for kc in kc_list:
            if kc not in self.params["theta_wins"]:
                continue
            
            tw_wins = np.array(self.params["theta_wins"][kc])
            tw_att  = np.array(self.params["theta_attempts"][kc])
            tf      = self.params["theta_fails"].get(kc, 0.0)
            
            mem += np.dot(tw_wins, np.log(1 + nwins.get(kc, np.zeros(5))))
            mem += np.dot(tw_att,  np.log(1 + natt.get(kc, np.zeros(5))))
            mem += tf * nfails.get(kc, 0)
        
        return mem
    
    def _compute_history(self, user_id, t_current, kc_list):
        """Calcule nwins, natt, nfails pour un élève."""
        df_user = self.df[
            (self.df["user_id"] == user_id) & 
            (self.df["timestamp"] < t_current)
        ].copy()
        
        if df_user.empty:
            return (
                {kc: np.zeros(5) for kc in kc_list},
                {kc: np.zeros(5) for kc in kc_list},
                {kc: 0 for kc in kc_list}
            )
        
        wins_timestamps     = defaultdict(list)
        attempts_timestamps = defaultdict(list)
        fails_count         = defaultdict(int)
        
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
        
        nwins, natt, nfails = {}, {}, {}
        for kc in kc_list:
            wins_ts = wins_timestamps[kc]
            att_ts  = attempts_timestamps[kc]
            
            wins_counts = []
            att_counts  = []
            for tw in self.TW_SECONDS:
                if tw == float("inf"):
                    wins_counts.append(len(wins_ts))
                    att_counts.append(len(att_ts))
                else:
                    wins_counts.append(sum(1 for t in wins_ts if (t_current - t) <= tw))
                    att_counts.append(sum(1 for t in att_ts if (t_current - t) <= tw))
            
            nwins[kc]  = np.array(wins_counts, dtype=float)
            natt[kc]   = np.array(att_counts, dtype=float)
            nfails[kc] = fails_count[kc]
        
        return nwins, natt, nfails
    
    def select_entities(self, entity_type, method="frequent", n=5, **kwargs):
        
        if entity_type == "users":
            all_entities = self.df["user_id"].unique()
            param_dict   = self.params["alpha_s"]
        else:  # items
            all_entities = self.df["item_id"].unique()
            param_dict   = self.params["delta_j"]
        
        if method == "frequent":
            col = "user_id" if entity_type == "users" else "item_id"
            return self.df[col].value_counts().head(n).index.tolist()
        
        elif method == "random":
            return np.random.choice(all_entities, size=min(n, len(all_entities)), 
                                    replace=False).tolist()
        
        elif method == "quantiles":
            # Sélectionner selon les quantiles d'un paramètre
            sorted_entities = sorted(param_dict.items(), key=lambda x: x[1])
            indices = [int(len(sorted_entities) * q) for q in np.linspace(0.1, 0.9, n)]
            return [sorted_entities[i][0] for i in indices]
        
        elif method == "cluster":
            # Clustering par ability ou difficulté
            ids    = list(param_dict.keys())
            values = np.array(list(param_dict.values())).reshape(-1, 1)
            
            n_clusters = kwargs.get("n_clusters", 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(values)
            
            clusters = {i: [] for i in range(n_clusters)}
            for eid, lab in zip(ids, labels):
                clusters[lab].append(eid)
            
            cluster_id = kwargs.get("cluster_id", 0)
            return clusters[cluster_id][:n]
        
        elif method == "by_param":
            # Sélectionner par valeur de paramètre (ex: ability entre 0 et 1)
            param = kwargs.get("param", "alpha_s" if entity_type == "users" else "delta_j")
            min_val = kwargs.get("min_val", -float("inf"))
            max_val = kwargs.get("max_val", float("inf"))
            
            filtered = [eid for eid, val in param_dict.items() 
                        if min_val <= val <= max_val]
            return filtered[:n]
        
        return list(all_entities)[:n]
   
    
    def plot_p_vs_ability(self, fixed_item=None, varied_users=None, 
                          user_selection="quantiles", n_users=5, t_current=None):
        
        # Sélection de l'item
        if fixed_item is None:
            fixed_item = self.df["item_id"].value_counts().index[0]
        
        df_item = self.df[self.df["item_id"] == fixed_item]
        if df_item.empty:
            print(f"[ERREUR] Item {fixed_item} introuvable")
            return
        
        kc_str  = df_item.iloc[0]["KC"]
        kc_list = str(kc_str).split("~~")
        delta   = self.params["delta_j"].get(fixed_item, 0.0)
        beta    = sum(self.params["beta_k"].get(kc, 0.0) for kc in kc_list)
        
        if t_current is None:
            t_current = df_item["timestamp"].max()
        
        # Sélection des élèves
        if varied_users is None:
            varied_users = self.select_entities("users", method=user_selection, n=n_users)
        
        # Tracer
        alphas = np.linspace(-3, 3, 300)
        fig, ax = plt.subplots(figsize=(11, 6))
        
        for i, user_id in enumerate(varied_users):
            mem = self.compute_memory(user_id, t_current, kc_list)
            logits = alphas - delta + beta + mem + self.params["intercept"]
            probs  = self.sigmoid(logits)
            
            real_alpha = self.params["alpha_s"].get(user_id, 0.0)
            color = COLORS[i % len(COLORS)]
            ax.plot(alphas, probs, color=color, lw=2.5, 
                    label=f"Élève {user_id} (mem={mem:.2f})", alpha=0.9)
            
        
        
        
        ax.axhline(0.5, color="#555566", lw=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Ability ", fontsize=11)
        ax.set_ylabel("P(correct)", fontsize=11)
        ax.set_title(f"P(correct) vs Ability — Item {fixed_item} fixé (δ={delta:.2f})", 
                     fontsize=12)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_p_vs_difficulty(self, fixed_user=None, varied_items=None,
                             item_selection="quantiles", n_items=5, t_current=None):
        
        # Sélection de l'élève
        if fixed_user is None:
            fixed_user = self.df["user_id"].value_counts().index[0]
        
        alpha = self.params["alpha_s"].get(fixed_user, 0.0)
        
        if t_current is None:
            df_user = self.df[self.df["user_id"] == fixed_user]
            if df_user.empty:
                print(f"[ERREUR] Élève {fixed_user} introuvable")
                return
            t_current = df_user["timestamp"].max()
        
        # Sélection des items
        if varied_items is None:
            varied_items = self.select_entities("items", method=item_selection, n=n_items)
        
        # Tracer
        deltas = np.linspace(-4, 4, 300)
        fig, ax = plt.subplots(figsize=(11, 6))
        
        for i, item_id in enumerate(varied_items):
            item_rows = self.df[self.df["item_id"] == item_id]
            if item_rows.empty:
                continue
            
            kc_str  = item_rows.iloc[0]["KC"]
            kc_list = str(kc_str).split("~~")
            beta    = sum(self.params["beta_k"].get(kc, 0.0) for kc in kc_list)
            mem     = self.compute_memory(fixed_user, t_current, kc_list)
            
            logits = alpha - deltas + beta + mem + self.params["intercept"]
            probs  = self.sigmoid(logits)
            
            real_delta = self.params["delta_j"].get(item_id, 0.0)
            color = COLORS[i % len(COLORS)]
            ax.plot(deltas, probs, color=color, lw=2.5, 
                    label=f"Item {item_id} (mem={mem:.2f})", alpha=0.9)
            
        """real_delta= self.params["delta_j"].get(fixed_user, 0.0)
        ax.axvline(real_delta, color=ORANGE, lw=2, linestyle="-", alpha=0.8,
                label=f"Difficulty student {fixed_user} (delta={real_delta:.2f})")"""
        ax.axhline(0.5, color="#555566", lw=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Difficulté δ", fontsize=11)
        ax.set_ylabel("P(correct)", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f"P(correct) vs Difficulté — Élève {fixed_user} fixé (α={alpha:.2f})", 
                     fontsize=12)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_p_vs_memory(self, fixed_item=None, fixed_ability_range=None,
                         n_students=10, t_current=None):
        
        # Sélection de l'item
        if fixed_item is None:
            fixed_item = self.df["item_id"].value_counts().index[0]
        
        df_item = self.df[self.df["item_id"] == fixed_item]
        if df_item.empty:
            print(f"[ERREUR] Item {fixed_item} introuvable")
            return
        
        kc_str  = df_item.iloc[0]["KC"]
        kc_list = str(kc_str).split("~~")
        delta   = self.params["delta_j"].get(fixed_item, 0.0)
        beta    = sum(self.params["beta_k"].get(kc, 0.0) for kc in kc_list)
        
        if t_current is None:
            t_current = df_item["timestamp"].max()
        
        # Sélectionner élèves avec ability similaire
        if fixed_ability_range is None:
            fixed_ability_range = (-0.3, 0.3)
        
        students = self.select_entities("users", method="by_param", n=100,
                                        param="alpha_s", 
                                        min_val=fixed_ability_range[0],
                                        max_val=fixed_ability_range[1])
        
        # Calculer (mémoire, proba) pour chaque élève
        data = []
        for user_id in students:
            if user_id not in df_item["user_id"].values:
                continue
            
            alpha = self.params["alpha_s"].get(user_id, 0.0)
            mem   = self.compute_memory(user_id, t_current, kc_list)
            logit = alpha - delta + beta + mem + self.params["intercept"]
            prob  = self.sigmoid(logit)
            
            data.append({"user_id": user_id, "alpha": alpha, "mem": mem, "prob": prob})
        
        if len(data) < 3:
            print(f"[ERREUR] Pas assez d'élèves avec α ∈ {fixed_ability_range}")
            return
        
        data = sorted(data, key=lambda x: x["mem"])[:n_students]
        
        # Tracer
        fig, ax = plt.subplots(figsize=(11, 6))
        
        mems  = [d["mem"] for d in data]
        probs = [d["prob"] for d in data]
        alphas_val = [d["alpha"] for d in data]
        
        scatter = ax.scatter(mems, probs, c=alphas_val, cmap="coolwarm", 
                             s=100, edgecolors="white", linewidths=1, zorder=5)
        ax.plot(mems, probs, color=BLUE, lw=2, alpha=0.4, zorder=3)
        
        for d in data:
            ax.annotate(f"{d['user_id']}", (d["mem"], d["prob"]), 
                        fontsize=7, ha="center", xytext=(0, 8), 
                        textcoords="offset points")
        
        plt.colorbar(scatter, ax=ax, label="Ability α")
        ax.axhline(0.5, color="#555566", lw=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Contribution mémoire", fontsize=11)
        ax.set_ylabel("P(correct)", fontsize=11)
        ax.set_title(
            f"P(correct) vs Mémoire — Item {fixed_item} (δ={delta:.2f})\n"
            f"Élèves avec α ∈ [{fixed_ability_range[0]:.1f}, {fixed_ability_range[1]:.1f}]",
            fontsize=12
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    

    
    def plot_ability_vs_difficulty_heatmap(self, fixed_memory=0.0, 
                                            alpha_range=(-2, 2), 
                                            delta_range=(-2, 2)):
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], 50)
        deltas = np.linspace(delta_range[0], delta_range[1], 50)
        
        A, D = np.meshgrid(alphas, deltas)
        logits = A - D + fixed_memory + self.params["intercept"]
        P = self.sigmoid(logits)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(A, D, P, levels=20, cmap="RdYlGn")
        ax.contour(A, D, P, levels=[0.5], colors="black", linewidths=2, linestyles="--")
        
        plt.colorbar(im, ax=ax, label="P(correct)")
        ax.set_xlabel("Ability αₛ", fontsize=11)
        ax.set_ylabel("Difficulté δⱼ", fontsize=11)
        ax.set_title(f"Heatmap P(correct) — Mémoire fixée à {fixed_memory:.2f}", 
                     fontsize=12)
        plt.tight_layout()
        plt.show()

    def plottetaWinsAttem(self,top_n_kc=8):
        params = self.params
        TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
        
        # Sélectionner top KC
        kc_magnitudes = {kc: np.sum(np.abs(params["theta_wins"][kc])) 
                        for kc in params["theta_wins"].keys()}
        top_kcs = sorted(kc_magnitudes.items(), key=lambda x: x[1], reverse=True)[:top_n_kc]
        top_kc_names = [kc for kc, _ in top_kcs]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        
        # ── A : θ_wins par fenêtre temporelle ────────────────────────
        ax_a = axes[0]
        x = np.arange(len(TW_LABELS))
        
        for i, kc in enumerate(top_kc_names):
            tw_wins = params["theta_wins"][kc]
            color = COLORS[i % len(COLORS)]
            ax_a.plot(x, tw_wins, marker="o", color=color, lw=2, ms=6,
                    label=kc[:30] + "..." if len(kc) > 30 else kc)
        
        ax_a.axhline(0, color="#555566", lw=1, linestyle="--", alpha=0.6)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(TW_LABELS)
        ax_a.set_xlabel("Fenêtre temporelle", fontsize=11)
        ax_a.set_ylabel("teta_wins", fontsize=11)
        ax_a.set_title("variation  teta_attempts", fontsize=11)
        ax_a.legend(fontsize=7, framealpha=0.9, loc="best")
        ax_a.grid(True, alpha=0.3)


        # ── A : θ_attempts par fenêtre temporelle ────────────────────────
        ax_a = axes[1]
        x = np.arange(len(TW_LABELS))
        
        for i, kc in enumerate(top_kc_names):
            tw_attemps = params["theta_attempts"][kc]
            color = COLORS[i % len(COLORS)]
            ax_a.plot(x, tw_attemps, marker="o", color=color, lw=2, ms=6,
                    label=kc[:30] + "..." if len(kc) > 30 else kc)
        
        ax_a.axhline(0, color="#555566", lw=1, linestyle="--", alpha=0.6)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(TW_LABELS)
        ax_a.set_xlabel("Fenêtre temporelle", fontsize=11)
        ax_a.set_ylabel("theta_attempts", fontsize=11)
        ax_a.set_title("Variation theta attempts", fontsize=11)
        ax_a.legend(fontsize=7, framealpha=0.9, loc="best")
        ax_a.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_theta_analysis(self, top_n_kc=8):
        
        params = self.params
        TW_LABELS = ["1h", "1j", "1sem", "1mois", "∞"]
        
        # Sélectionner top KC
        kc_magnitudes = {kc: np.sum(np.abs(params["theta_wins"][kc])) 
                        for kc in params["theta_wins"].keys()}
        top_kcs = sorted(kc_magnitudes.items(), key=lambda x: x[1], reverse=True)[:top_n_kc]
        top_kc_names = [kc for kc, _ in top_kcs]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ── A : θ_wins par fenêtre temporelle ────────────────────────
        ax_a = axes[0, 0]
        x = np.arange(len(TW_LABELS))
        
        for i, kc in enumerate(top_kc_names):
            tw_wins = params["theta_wins"][kc]
            color = COLORS[i % len(COLORS)]
            ax_a.plot(x, tw_wins, marker="o", color=color, lw=2, ms=6,
                    label=kc[:30] + "..." if len(kc) > 30 else kc)
        
        ax_a.axhline(0, color="#555566", lw=1, linestyle="--", alpha=0.6)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(TW_LABELS)
        ax_a.set_xlabel("Fenêtre temporelle", fontsize=11)
        ax_a.set_ylabel("teta_wins", fontsize=11)
        ax_a.set_title("Décroissance teta_wins — Courbes d'oubli", fontsize=11)
        ax_a.legend(fontsize=7, framealpha=0.9, loc="best")
        ax_a.grid(True, alpha=0.3)
        
        # ── B : Ratio 1h / ∞ (vitesse d'oubli) ───────────────────────
        ax_b = axes[0, 1]
        
        ratios = []
        labels = []
        for kc in top_kc_names:
            tw_wins = params["theta_wins"][kc]
            ratio = tw_wins[0] / tw_wins[-1] if tw_wins[-1] != 0 else 0
            ratios.append(ratio)
            labels.append(kc[:25] + "..." if len(kc) > 25 else kc)
        
        bars = ax_b.barh(labels, ratios, color=COLORS[:len(labels)])
        ax_b.axvline(5, color=ORANGE, lw=2, linestyle="--", alpha=0.8, 
                    label="Seuil oubli rapide")
        ax_b.set_xlabel("Ratio θ_wins[1h] / θ_wins[∞]", fontsize=11)
        ax_b.set_title("Vitesse d'oubli par KC", fontsize=11)
        ax_b.legend(fontsize=9)
        ax_b.grid(True, alpha=0.3, axis="x")
        
        # ── C : θ_fails vs β_k (pénalité échecs vs facilité) ─────────
        ax_c = axes[1, 0]
        
        betas = []
        fails = []
        kc_names = []
        
        for kc in top_kc_names:
            beta = params["beta_k"].get(kc, 0)
            fail = params["theta_fails"].get(kc, 0)
            betas.append(beta)
            fails.append(fail)
            kc_names.append(kc[:20])
        
        scatter = ax_c.scatter(betas, fails, s=150, c=range(len(betas)), 
                            cmap="viridis", edgecolors="white", linewidths=1.5,
                            zorder=5)
        
        for i, (b, f, name) in enumerate(zip(betas, fails, kc_names)):
            ax_c.annotate(name, (b, f), fontsize=7, ha="left",
                        xytext=(5, 0), textcoords="offset points")
        
        ax_c.axhline(0, color="#555566", lw=1, linestyle="-", alpha=0.6)
        ax_c.axvline(0, color="#555566", lw=1, linestyle="-", alpha=0.6)
        ax_c.set_xlabel("βₖ (easiness)", fontsize=11)
        ax_c.set_ylabel("θ_fails (pénalité échecs)", fontsize=11)
        ax_c.set_title("Facilité vs Pénalité échecs", fontsize=11)
        ax_c.grid(True, alpha=0.3)
        
        # Quadrants
        ax_c.text(max(betas)*0.8, min(fails)*0.8, "Facile +\nÉchecs pénalisants",
                fontsize=8, ha="right", va="bottom", alpha=0.5)
        ax_c.text(min(betas)*0.8, min(fails)*0.8, "Difficile +\nÉchecs pénalisants",
                fontsize=8, ha="left", va="bottom", alpha=0.5)
        
        # ── D : Contribution moyenne par fenêtre (global) ────────────
        ax_d = axes[1, 1]
        
        avg_wins = np.zeros(5)
        avg_att  = np.zeros(5)
        
        for kc in params["theta_wins"].keys():
            avg_wins += np.array(params["theta_wins"][kc])
            avg_att  += np.array(params["theta_attempts"].get(kc, np.zeros(5)))
        
        avg_wins /= len(params["theta_wins"])
        avg_att  /= len(params["theta_wins"])
        
        width = 0.35
        x = np.arange(len(TW_LABELS))
        
        ax_d.bar(x - width/2, avg_wins, width, label="θ_wins (succès)", 
                color=GREEN, alpha=0.8)
        ax_d.bar(x + width/2, avg_att, width, label="θ_attempts (tentatives)", 
                color=BLUE, alpha=0.8)
        
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(TW_LABELS)
        ax_d.set_xlabel("Fenêtre temporelle", fontsize=11)
        ax_d.set_ylabel("Coefficient moyen", fontsize=11)
        ax_d.set_title("Contribution mémoire moyenne — Tous les KC", fontsize=11)
        ax_d.legend(fontsize=9)
        ax_d.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.show()
    
    