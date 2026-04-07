import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)


import numpy as np
import pandas as pd
import os
from scipy import sparse


N_STUDENTS           = 10
N_ITEMS              = 100  
N_KC                 = 20  
N_DAYS               = 30
INTERACTIONS_PER_DAY = 10
N_REVIEWS            = 10  
SEED                 = 42
np.random.seed(SEED)

T_START = int(pd.Timestamp("2024-01-01").timestamp())
DAY_SEC = 3600 * 24


Q_mat = np.zeros((N_ITEMS, N_KC), dtype=int)
for item in range(N_ITEMS):
    n_kc_item = np.random.choice([1, 2], p=[0.6, 0.4])
    kcs = np.random.choice(N_KC, size=n_kc_item, replace=False)
    Q_mat[item, kcs] = 1


student_ability = np.random.normal(0.0, 0.5, N_STUDENTS)  # alpha_s

item_difficulty = np.random.normal(0.0, 0.5, N_ITEMS)     # delta_j

kc_skill = np.random.normal(0.0, 0.3, N_KC)               # beta_k

rows = []
inter_id = 0

for sid in range(N_STUDENTS):
    
    # pool d'items RESTREINT → répétition garantie
    # chaque étudiant pioche dans les N_ITEMS disponibles avec remise
    kc_attempts = np.zeros(N_KC)
    kc_wins     = np.zeros(N_KC)
    
    for day in range(N_DAYS):
        if day % 7 >= 5:  # pas le weekend
            continue
        
        t_day = T_START + day * DAY_SEC + np.random.randint(8*3600, 10*3600)
        
        for interaction in range(INTERACTIONS_PER_DAY):
            
            # choisir un item AU HASARD parmi les N_ITEMS → répétition possible
            item_id = np.random.randint(0, N_ITEMS)
            
            kcs_item = np.where(Q_mat[item_id] == 1)[0]
            t = t_day + interaction * 300 + np.random.randint(-30, 30)
            
            logit = (student_ability[sid]
                     - item_difficulty[item_id]
                     + sum(kc_skill[kc] for kc in kcs_item)
                     + 0.1 * sum(np.log(1 + kc_wins[kc]) for kc in kcs_item))
            
            prob    = 1 / (1 + np.exp(-logit))
            correct = int(np.random.random() < prob)
            
            for kc in kcs_item:
                kc_attempts[kc] += 1
                if correct:
                    kc_wins[kc] += 1
            
            kc_str = "~~".join([f"KC_{kc}" for kc in kcs_item])
            rows.append({
                "user_id":   sid,
                "item_id":   item_id,
                "KC":        kc_str,
                "timestamp": t,
                "correct":   correct,
                "inter_id":  inter_id
            })
            inter_id += 1

df = pd.DataFrame(rows)

OUT_FOLDER = os.path.join("data", "simulated")
os.makedirs(OUT_FOLDER, exist_ok=True)

df.to_csv(os.path.join(OUT_FOLDER, "preprocessed_data_simulated.csv"), index=False)
sparse.save_npz(os.path.join(OUT_FOLDER, "q_mat_simulated.npz"),
                sparse.csr_matrix(Q_mat))

# ajoute ces vérifications à la fin de ton script de génération
print("\n=== VÉRIFICATIONS DÉTAILLÉES ===")

# 1. répétitions par (étudiant, item)
repeats = df.groupby(["user_id", "item_id"]).size()
print(f"\nRépétitions par (étudiant, item) :")
print(f"  moyenne : {repeats.mean():.2f}")
print(f"  min     : {repeats.min()}")
print(f"  max     : {repeats.max()}")
print(f"  items vus 1 seule fois : {(repeats==1).mean()*100:.1f}%")
print(f"  items vus 3+ fois      : {(repeats>=3).mean()*100:.1f}%")

# 2. pour un seul étudiant : overlap train/test
df_s0 = df[df["user_id"]==0].sort_values("timestamp")
n = len(df_s0)
split = int(n * 0.8)
train_items = set(df_s0.iloc[:split]["item_id"])
test_items  = set(df_s0.iloc[split:]["item_id"])
overlap = train_items & test_items
print(f"\nÉtudiant 0 — overlap items train/test :")
print(f"  items en train : {len(train_items)}")
print(f"  items en test  : {len(test_items)}")
print(f"  items communs  : {len(overlap)} ({len(overlap)/len(test_items)*100:.1f}% du test)")

# 3. interactions par jour actif
df["day"] = (df["timestamp"] - df["timestamp"].min()) // (3600*24)
interactions_per_day = df.groupby(["user_id", "day"]).size()
print(f"\nInteractions par jour actif :")
print(f"  moyenne : {interactions_per_day.mean():.2f}")
print(f"  min     : {interactions_per_day.min()}")