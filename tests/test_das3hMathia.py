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
import src.graphics.PlotOutills as Plot
import src.graphics.das3hviz as Vis
import joblib
import time
import seaborn as sns
from utils.this_queue import OurQueue
from collections import defaultdict

NAME_FOLDER="Mathiadata"
DATA_FOLDER=os.path.join("data",NAME_FOLDER)
N_STUDENTS = 25351 # Number of students to use real user = 1146 , item =19355
MIN_INTERACTIONS = 30
MODEL_C = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]
N_TIME_WINDOWS = 5

def load_student_model(data_folder: str,mininteractions: int = 30,n_students: int = 100):
    
    print("!!!!!!!!!!!!!!!!Loading student model !!!!!!!!!!!!")
    pathMathiadata = os.path.join(data_folder, "..", NAME_FOLDER, "data.csv")
    stdmodel :SD.StudentDATA = SD.Mathiadata(pathMathiadata,seed=42)
    df ,Q= stdmodel.loadData(Display=True, min_intercation=mininteractions, n_students= n_students)
    
    return stdmodel, df, Q
if __name__ == "__main__":
    stdmodel,df,Q = load_student_model(DATA_FOLDER, MIN_INTERACTIONS, N_STUDENTS)
    print("!!!!Done!!!!!")