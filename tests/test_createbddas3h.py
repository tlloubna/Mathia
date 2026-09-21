import os
import sys
from pathlib import Path

# Add parent directory to path
extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
import pandas as pd
from scipy import sparse

import src.datamodel.Studentdata as SD
import matplotlib.pyplot as plt
import src.Process.DAS3H as DAS3H
import src.datamodel.Historydata as HIS
import joblib
from utils.this_queue import OurQueue
from collections import defaultdict
import time
from sklearn.calibration import calibration_curve

from sklearn.metrics import brier_score_loss, log_loss
NAME_FOLDER="Mathiadata_v2" #algebra =574,item 1084
DATA_FOLDER = os.path.join("data",NAME_FOLDER)
N_STUDENTS =100000# Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30
MODEL_C = 0.01  # Regularization parameter
N_TIME_WINDOWS = 5

def prepare_featuresAlpha( data_folder,df,q_matrix: np.ndarray, stdmodel: SD.StudentDATA ):
    print("!!!!!!!!!!!!!!!!Preparing history !!!!!!!!!!!!")
    his = HIS.HistoryDATA(stdmodel=stdmodel)
    X, user_ids, item_ids, listKC = his.ComputeHistoryFeaturesALPHASK(Q_mat=q_matrix, df=df)
    #save X to npz file
    sparse.save_npz(os.path.join(data_folder, f"history_features_Alpha{N_STUDENTS}std.npz"), sparse.csr_matrix(X))
    np.savez(os.path.join(data_folder, f"history_metadata_Alpha{N_STUDENTS}std.npz"), user_ids=user_ids, item_ids=item_ids, kc_list=listKC)
    return X, user_ids, item_ids, listKC

if __name__ == "__main__":
    timetoexeucte=1#Time to execute

    if timetoexeucte==1: 
        df=pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
        q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
        start=time.time()
        X_alpha, user_ids, item_ids, kc_list = prepare_featuresAlpha(DATA_FOLDER,df,q_matrix, stdmodel=None)
        end=time.time()
        print(f"Time to prepare features: {end - start:.2f} seconds")
    print("!!!!!!!!!done!!!!!!!!!!!!!")