import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)
import numpy as np
from scipy import sparse
import pandas as pd 
import matplotlib.pyplot as plt 



def prepare_assistments12(min_interactions_per_user, remove_nan_skills, verbose,pathassist):
	df = pd.read_csv(pathassist, usecols=["user_id", "problem_id", "skill_id", "correct", "start_time"])
	if verbose:
		initial_shape = df.shape[0]
		print("Opened ASSISTments 2012 data. Output: {} samples.".format(initial_shape))
	
	df["timestamp"] = df["start_time"]
	df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
	
	if remove_nan_skills:
		df = df[~df["skill_id"].isnull()]
		if verbose:
			print("Removed {} samples with NaN skills.".format(df.shape[0]-initial_shape))
			initial_shape = df.shape[0]
	else:
		df.loc[df["skill_id"].isnull(), "skill_id"] = -1

	df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
	if verbose:
		print("Removed {} samples with non-binary outcomes.".format(df.shape[0]-initial_shape))
		initial_shape = df.shape[0]
	df['correct'] = df['correct'].astype(np.int32) # Cast outcome as int32

	df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
	if verbose:
		print('Removed {} samples (users with less than {} interactions).'.format(df.shape[0]-initial_shape, min_interactions_per_user))
		initial_shape = df.shape[0]

	df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
	df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
	df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]
	
	
	Q_mat = np.zeros((df["item_id"].nunique(), df["skill_id"].nunique()))
	item_skill = np.array(df[["item_id", "skill_id"]])
	for i in range(len(item_skill)):
		Q_mat[item_skill[i,0],item_skill[i,1]] = 1
	if verbose:
		print("Computed q-matrix. Shape: {}.".format(Q_mat.shape))

	#df = df[['user_id', 'item_id', 'timestamp', 'correct', "inter_id"]]
	df = df[['user_id', 'item_id','skill_id', 'timestamp', 'correct']]
	# Remove potential duplicates
	df.drop_duplicates(inplace=True)
	if verbose:
		print("Removed {} duplicated samples.".format(df.shape[0] - initial_shape))
		initial_shape = df.shape[0]

	df.sort_values(by="timestamp", inplace=True)
	df.reset_index(inplace=True, drop=True)
	print("Data preprocessing done. Final output: {} samples.".format((df.shape[0])))
	# Save data
	
	return df, Q_mat



if __name__=="__main__":
    path_assist="/home/loubna/Code Projet Mathia/Mathia/data/ASSISTments13_12/ASSIST13_12.csv"
    df, Q_mat = prepare_assistments12(min_interactions_per_user=30, remove_nan_skills=True, verbose=True, pathassist=path_assist)
    
    df["inter_id"] = df.index
    
    df["KC"] = df["skill_id"]
    df["KC"] = "KC_" + df["skill_id"].astype(str)
	
    OUT_FOLDER = os.path.join("data", "ASSISTments13_12")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    
    n_students = df["user_id"].nunique()
    df.to_csv(os.path.join(OUT_FOLDER, f"preprocessed_data_{n_students}std.csv"), index=False)
    sparse.save_npz(os.path.join(OUT_FOLDER, f"q_mat_{n_students}std.npz"), sparse.csr_matrix(Q_mat))
    
    print(f"Saved: {n_students} students, {df['item_id'].nunique()} items, {Q_mat.shape[1]} KCs")
	
    