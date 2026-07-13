import os
import sys

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import time
import numpy as np
import pandas as pd
from scipy import sparse
import src.datamodel.Historydata as HIS


# Choisis le(s) dataset(s) à régénérer. Commence par MATHIA seul.
DATASETS = [
    #("Mathiadata", 25351),
    ("ASSISTments13_12", 15698),
     #("bridge_algebra06", 1146),   # ATTENTION : ~19355 items -> très lourd
    # ("algebra05", 574),           # ATTENTION : ~1084 items
]

N_TIME_WINDOWS = 5


def regenerate_with_item_blocks(folder, n_students, n_tw=5):
    data_folder = os.path.join("data", folder)

    # Charge les données déjà prétraitées et la Q-matrix existantes
    df = pd.read_csv(os.path.join(data_folder, f"preprocessed_data_{n_students}std.csv"))
    q_matrix = sparse.load_npz(
        os.path.join(data_folder, f"q_mat_{n_students}std.npz")
    ).toarray()

    print(f"\n=== {folder} ===")
    print(f"  interactions : {len(df)}")
    print(f"  items (Q)    : {q_matrix.shape[0]}")
    print(f"  CC (Q)       : {q_matrix.shape[1]}")
    print(f"  colonnes item ajoutées : {q_matrix.shape[0] * n_tw * 2} "
          f"(wins_item + attempts_item)")

    his = HIS.HistoryDATA(stdmodel=None)
    his.TimeWindow = his.TimeWindow[:n_tw] if len(his.TimeWindow) >= n_tw else his.TimeWindow
    his.n_tw = len(his.TimeWindow)

    start = time.time()
    X, user_ids, item_ids, listKC = his.ComputeHistoryFeaturesTWKC_plusItems(
        Q_mat=q_matrix, df=df
    )
    elapsed = time.time() - start
    print(f"  -> X shape : {X.shape}  ({elapsed:.1f} s)")

    # Sauvegarde sous un nom DISTINCT pour ne PAS écraser tes matrices actuelles
    sparse.save_npz(
        os.path.join(data_folder, f"history_features_DASH_{n_students}std.npz"),
        sparse.csr_matrix(X)
    )
    np.savez(
        os.path.join(data_folder, f"history_metadata_DASH_{n_students}std.npz"),
        user_ids=user_ids, item_ids=item_ids, kc_list=listKC
    )
    print(f"  Sauvegardé : history_features_DASH_{n_students}std.npz")
    return X


if __name__ == "__main__":
    for folder, n_students in DATASETS:
        try:
            regenerate_with_item_blocks(folder, n_students, n_tw=N_TIME_WINDOWS)
        except Exception as e:
            print(f"[!] Échec sur {folder} : {e}")
    print("\n!!! Régénération terminée !!!")