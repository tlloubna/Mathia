import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import pandas as pd 
from scipy import sparse
import numpy as np 
FOLDER = "bridge_algebra06"
data_folder = os.path.join("data", FOLDER)
path_dash = "data/algebra05/preprocessed_data_574std.csv"
qmpath= "/home/loubna/Code_Projet_Mathia/Mathia/data/algebra05/q_mat_574std.npz"

df=pd.read_csv(path_dash)
Q_mat=sparse.load_npz(qmpath).toarray()

kc_per_item = {i: set(np.where(Q_mat[i]==1)[0]) for i in range(Q_mat.shape[0])}
cov = df.sort_values("timestamp").groupby("user_id").head(100).groupby("user_id")["item_id"].apply(
    lambda s: len(set().union(*[kc_per_item[i] for i in s]))).describe()
print(cov, "\n n_kc total =", Q_mat.shape[1])
