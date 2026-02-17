import src.datamodel.Studentdata as STD
import numpy as np 
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from utils.this_queue import OurQueue
from collections import defaultdict, Counter
TIME_WINDOWS = [3600, 86400, 604800, 2592000, float("inf")]
NB_OF_TIME_WINDOWS = len(TIME_WINDOWS)

class HistoryDATA:
    def __init__(self,stdmodel:STD.StudentDATA=None,TimeWindow:list=[3600,86400,604800,float("inf")]):
        self.stdmodel=stdmodel
        self.TimeWindow:list=TimeWindow
        


    def ComputeHistoryFeaturesTWKC(self, Q_mat, df):

        # Construire dict_q_mat
        dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)

        #  Initialiser X
        X = {
            "skills": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "attempts": sparse.csr_matrix(np.empty((0, Q_mat.shape[1] * NB_OF_TIME_WINDOWS))),
            "wins": sparse.csr_matrix(np.empty((0, Q_mat.shape[1] * NB_OF_TIME_WINDOWS))),
            "fails": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "df": np.empty((0, 5))
        }

        #  Files glissantes
        q = defaultdict(lambda: OurQueue())

        #  Boucle par élève
        for stud_id in df["user_id"].unique():

            df_stud = df[df["user_id"] == stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]]
            df_stud = df_stud.sort_values("timestamp").to_numpy()

            X["df"] = np.vstack((X["df"], df_stud))

            # Skills
            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"], sparse.csr_matrix(skills_temp)])

            # Attempts tw_kc
            attempts = np.zeros((df_stud.shape[0], Q_mat.shape[1] * NB_OF_TIME_WINDOWS))
            for l, (item_id, t) in enumerate(zip(df_stud[:, 1], df_stud[:, 2])):
                for kc in dict_q_mat[item_id]:
                    attempts[l, kc*NB_OF_TIME_WINDOWS:(kc+1)*NB_OF_TIME_WINDOWS] = np.log(
                        1 + np.array(q[stud_id, kc].get_counters(t))
                    )
                    q[stud_id, kc].push(t)
            X["attempts"] = sparse.vstack([X["attempts"], sparse.csr_matrix(attempts)])

            # Wins tw_kc
            wins = np.zeros((df_stud.shape[0], Q_mat.shape[1] * NB_OF_TIME_WINDOWS))
            for l, (item_id, t, correct) in enumerate(zip(df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                for kc in dict_q_mat[item_id]:
                    wins[l, kc*NB_OF_TIME_WINDOWS:(kc+1)*NB_OF_TIME_WINDOWS] = np.log(
                        1 + np.array(q[stud_id, kc, "correct"].get_counters(t))
                    )
                    if correct:
                        q[stud_id, kc, "correct"].push(t)
            X["wins"] = sparse.vstack([X["wins"], sparse.csr_matrix(wins)])

            # Fails
            fails = np.multiply(
                np.cumsum(
                    np.multiply(
                        np.vstack((np.zeros(skills_temp.shape[1]), skills_temp)),
                        np.hstack(([0], 1 - df_stud[:, 3])).reshape(-1, 1)
                    ), axis=0
                )[:-1],
                skills_temp
            )
            X["fails"] = sparse.vstack([X["fails"], sparse.csr_matrix(fails)])

        # One-hot users/items
        enc_users = OneHotEncoder()
        enc_items = OneHotEncoder()
        X["users"] = enc_users.fit_transform(X["df"][:, 0].reshape(-1, 1))
        X["items"] = enc_items.fit_transform(X["df"][:, 1].reshape(-1, 1))
        self.user_ids = enc_users.categories_[0].tolist()  # ← sauvegarde l'ordre
        self.item_ids = enc_items.categories_[0].tolist()  # ← sauvegarde l'ordre

        # Construction finale
        sparse_df = sparse.hstack([
            sparse.csr_matrix(X["df"]),
            X["users"],
            X["items"],
            X["skills"],
            X["wins"],
            X["fails"],
            X["attempts"]
        ]).tocsr()

        return sparse_df
