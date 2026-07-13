import src.datamodel.Studentdata as STD
import numpy as np 
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from utils.this_queue import OurQueue
from collections import defaultdict, Counter
TIME_WINDOWS_DEFAULT = [3600, 86400, 604800, 2592000, float("inf")]
#NB_OF_TIME_WINDOWS = len(TIME_WINDOWS)

class HistoryDATA:
    def __init__(self,stdmodel:STD.StudentDATA=None,TimeWindow:list=None):
        self.stdmodel=stdmodel
        self.TimeWindow:list=TimeWindow if TimeWindow is not None else TIME_WINDOWS_DEFAULT
        self.n_tw=len(self.TimeWindow)

    def make_queue(self):
        return OurQueue(window_lengths=self.TimeWindow)
    def ComputeHistoryFeaturesTWKC(self, Q_mat, df, vocab_users=None, vocab_items=None):
        n_tw=self.n_tw
        # Construire dict_q_mat
        dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)

        #  Initialiser X
        X = {
            "skills": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "attempts": sparse.csr_matrix(np.empty((0, Q_mat.shape[1] * n_tw))),
            "wins": sparse.csr_matrix(np.empty((0, Q_mat.shape[1] * n_tw))),
            "fails": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "df": np.empty((0, 5))
        }

        #  Files glissantes
        q = defaultdict(self.make_queue)

        #  Boucle par élève
        for idx,stud_id in enumerate(df["user_id"].unique()):
            print("Stud_id:",idx+1,"/",len(df["user_id"].unique()))
            
            df_stud = df[df["user_id"] == stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]]
            df_stud = df_stud.sort_values("timestamp").to_numpy()

            if df_stud.shape[0] == 0:
                print(f" Élève {stud_id} n'a AUCUNE interaction dans ce DataFrame !")
                continue
            X["df"] = np.vstack((X["df"], df_stud))

            # Skills
            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"], sparse.csr_matrix(skills_temp)])

            # Attempts tw_kc
            attempts = np.zeros((df_stud.shape[0], Q_mat.shape[1] * n_tw))
            for l, (item_id, t) in enumerate(zip(df_stud[:, 1], df_stud[:, 2])):
                for kc in dict_q_mat[item_id]:
                    attempts[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc].get_counters(t))
                    )
                    q[stud_id, kc].push(t)
            X["attempts"] = sparse.vstack([X["attempts"], sparse.csr_matrix(attempts)])

            # Wins tw_kc
            wins = np.zeros((df_stud.shape[0], Q_mat.shape[1] * n_tw))
            for l, (item_id, t, correct) in enumerate(zip(df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                for kc in dict_q_mat[item_id]:
                    wins[l, kc*n_tw:(kc+1)*n_tw] = np.log(
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
        enc_users = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        enc_items = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        
        if vocab_users is not None:
            enc_users.fit(np.array(vocab_users).reshape(-1, 1))
        else:
            enc_users.fit(X["df"][:, 0].reshape(-1, 1))
        
        if vocab_items is not None:
            enc_items.fit(np.array(vocab_items).reshape(-1, 1))
        else:
            enc_items.fit(X["df"][:, 1].reshape(-1, 1))

        X["users"] = enc_users.transform(X["df"][:, 0].reshape(-1, 1))
        X["items"] = enc_items.transform(X["df"][:, 1].reshape(-1, 1))
        self.user_ids = enc_users.categories_[0].tolist()  
        self.item_ids = enc_items.categories_[0].tolist()  
        listOfKC = []
        for kc_raw in df["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)
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
       
        return sparse_df, self.user_ids, self.item_ids, listOfKC
    
    def ComputeHistoryFeaturesTWKC_plusItems(self, Q_mat, df, vocab_users=None, vocab_items=None):
        n_tw = self.n_tw
        n_items = Q_mat.shape[0]
        n_kc = Q_mat.shape[1]
        # dict item -> set de CC
        dict_q_mat = {i: set() for i in range(n_items)}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)

        X = {
            "skills":        sparse.csr_matrix(np.empty((0, n_kc))),
            "attempts":      sparse.csr_matrix(np.empty((0, n_kc * n_tw))),   # par CC
            "wins":          sparse.csr_matrix(np.empty((0, n_kc * n_tw))),   # par CC
            "fails":         sparse.csr_matrix(np.empty((0, n_kc))),
            "attempts_item":  sparse.csr_matrix(np.empty((0, n_tw))),  # NEW : par item
            "wins_item":     sparse.csr_matrix(np.empty((0, n_tw))),  # NEW : par item
            "df":            np.empty((0, 5))
        }

        q = defaultdict(self.make_queue)

        for idx, stud_id in enumerate(df["user_id"].unique()):
            print("Stud_id:", idx + 1, "/", len(df["user_id"].unique()))

            df_stud = df[df["user_id"] == stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]]
            df_stud = df_stud.sort_values("timestamp").to_numpy()
            if df_stud.shape[0] == 0:
                continue
            X["df"] = np.vstack((X["df"], df_stud))

            # Skills (inchangé)
            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"], sparse.csr_matrix(skills_temp)])

            # Attempts par CC (inchangé)
            attempts = np.zeros((df_stud.shape[0], n_kc * n_tw))
            for l, (item_id, t) in enumerate(zip(df_stud[:, 1], df_stud[:, 2])):
                for kc in dict_q_mat[item_id]:
                    attempts[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc].get_counters(t))
                    )
                    q[stud_id, kc].push(t)
            X["attempts"] = sparse.vstack([X["attempts"], sparse.csr_matrix(attempts)])

            # Wins par CC (inchangé)
            wins = np.zeros((df_stud.shape[0], n_kc * n_tw))
            for l, (item_id, t, correct) in enumerate(zip(df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                for kc in dict_q_mat[item_id]:
                    wins[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc, "correct"].get_counters(t))
                    )
                    if correct:
                        q[stud_id, kc, "correct"].push(t)
            X["wins"] = sparse.vstack([X["wins"], sparse.csr_matrix(wins)])

            # ---- NEW : Attempts par ITEM (fenêtres temporelles) ----
            attempts_item = np.zeros((df_stud.shape[0], n_tw))
            for l, (item_id, t) in enumerate(zip(df_stud[:, 1], df_stud[:, 2])):
                item_id = int(item_id)
                attempts_item[l] = np.log(1 + np.array(q[stud_id, "item", item_id].get_counters(t)))
                q[stud_id, "item", item_id].push(t)
            X["attempts_item"] = sparse.vstack([X["attempts_item"], sparse.csr_matrix(attempts_item)])

        
            # ---- NEW : Wins par ITEM (fenêtres temporelles) ----
            wins_item = np.zeros((df_stud.shape[0], n_tw))
            for l, (item_id, t, correct) in enumerate(zip(df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                item_id = int(item_id)
                wins_item[l] = np.log(1 + np.array(q[stud_id, "item", item_id, "correct"].get_counters(t)))
                if correct:
                    q[stud_id, "item", item_id, "correct"].push(t)
            X["wins_item"] = sparse.vstack([X["wins_item"], sparse.csr_matrix(wins_item)])
               # Fails (inchangé)
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

        # One-hot users/items (inchangé)
        enc_users = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        enc_items = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        enc_users.fit(np.array(vocab_users).reshape(-1, 1) if vocab_users is not None else X["df"][:, 0].reshape(-1, 1))
        enc_items.fit(np.array(vocab_items).reshape(-1, 1) if vocab_items is not None else X["df"][:, 1].reshape(-1, 1))
        X["users"] = enc_users.transform(X["df"][:, 0].reshape(-1, 1))
        X["items"] = enc_items.transform(X["df"][:, 1].reshape(-1, 1))
        self.user_ids = enc_users.categories_[0].tolist()
        self.item_ids = enc_items.categories_[0].tolist()

        listOfKC = []
        for kc_raw in df["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)

        # Construction finale : on AJOUTE wins_item et attempts_item à la fin
        sparse_df = sparse.hstack([
            sparse.csr_matrix(X["df"]),
            X["users"],
            X["items"],
            X["skills"],
            X["wins"],
            X["fails"],
            X["attempts"],
            X["wins_item"],        # NEW
            X["attempts_item"],    # NEW
        ]).tocsr()

        return sparse_df, self.user_ids, self.item_ids, listOfKC
    
    def ComputeHistoryFeaturesALPHASK(self, Q_mat, df):
        n_tw=self.n_tw
        # Construire dict_q_mat
        dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)

        n_kc    = Q_mat.shape[1]
        n_users = df["user_id"].nunique()
        n_pairs = n_users * n_kc  # nb colonnes pour alpha_{s,k}

        user_ids_list = sorted(df["user_id"].unique())
        user_to_idx   = {u: i for i, u in enumerate(user_ids_list)}

        X = {
            "users_kc": sparse.csr_matrix(np.empty((0, n_pairs))),  
            "skills":   sparse.csr_matrix(np.empty((0, n_kc))),
            "attempts": sparse.csr_matrix(np.empty((0, n_kc * n_tw))),
            "wins":     sparse.csr_matrix(np.empty((0, n_kc * n_tw))),
            "fails":    sparse.csr_matrix(np.empty((0, n_kc))),
            "df":       np.empty((0, 5))
        }

        q = defaultdict(self.make_queue)

        # Boucle par élève
        for idx, stud_id in enumerate(df["user_id"].unique()):
            print(f"Stud_id: {idx+1}/{n_users} ({(idx+1)/n_users*100:.0f}%)")

            df_stud = df[df["user_id"] == stud_id][
                ["user_id", "item_id", "timestamp", "correct", "inter_id"]
            ].sort_values("timestamp").to_numpy()

            X["df"] = np.vstack((X["df"], df_stud))

            user_idx   = user_to_idx[stud_id]
            n_inter    = df_stud.shape[0]

            # Skills
            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"], sparse.csr_matrix(skills_temp)])

            attempts  = np.zeros((n_inter, n_kc * n_tw))
            wins      = np.zeros((n_inter, n_kc * n_tw))

            users_kc_local = np.zeros((n_inter, n_kc))  # bloc local (n_inter × n_kc)
                
            for l, (item_id, t, correct) in enumerate(zip(
                    df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                for kc in dict_q_mat[item_id]:
                    #Attempts
                    attempts[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc].get_counters(t))
                    )
                    #Wins
                    wins[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc, "correct"].get_counters(t))
                    )
                    #user_kc
                    users_kc_local[l, kc] = 1.0
                    q[stud_id, kc].push(t)
                    if correct:
                        q[stud_id, kc, "correct"].push(t)

            X["attempts"] = sparse.vstack([X["attempts"], sparse.csr_matrix(attempts)])
            X["wins"]     = sparse.vstack([X["wins"],     sparse.csr_matrix(wins)])
            users_kc_row = sparse.hstack([
                sparse.csr_matrix((n_inter, user_idx * n_kc)),           # zéros avant
                sparse.csr_matrix(users_kc_local),                        # bloc local
                sparse.csr_matrix((n_inter, (n_users - user_idx - 1) * n_kc))  # zéros après
            ])
            X["users_kc"] = sparse.vstack([X["users_kc"], users_kc_row])

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
        enc_items = OneHotEncoder()
        X["items"] = enc_items.fit_transform(X["df"][:, 1].reshape(-1, 1))
        self.item_ids = enc_items.categories_[0].tolist()
        self.user_ids = user_ids_list

        listOfKC = []
        for kc_raw in df["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)
        sparse_df = sparse.hstack([
            sparse.csr_matrix(X["df"]),
            X["users_kc"],   
            X["items"],
            X["skills"],
            X["wins"],
            X["fails"],
            X["attempts"]
        ]).tocsr()

        return sparse_df, self.user_ids, self.item_ids, listOfKC
        

    def ComputeHistoryFeaturesAlphabestInter(self, Q_mat, df, min_kc_interactions=1000):
        """
        min_kc_interactions : seuil minimum d'interactions par KC
                            pour être inclus dans alpha_sk
        """
        # Construire dict_q_mat
        n_tw=self.n_tw
        dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)

       
        df_exploded = df.copy()
        df_exploded["KC"] = df_exploded["KC"].str.split("~~")
        df_exploded = df_exploded.explode("KC")
        kc_counts = df_exploded["KC"].value_counts()

        top_kcs_names = set(kc_counts[kc_counts >= min_kc_interactions].index)
        self.top_kcs = sorted(top_kcs_names)  
        print(f"KCs retenus pour alpha_sk : {len(top_kcs_names)} / "
            f"{len(kc_counts)} (seuil={min_kc_interactions})")

        listOfKC = []
        for kc_raw in df["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)

        kc_to_idx  = {kc: i for i, kc in enumerate(listOfKC)}
        top_kc_idx = sorted([kc_to_idx[kc] for kc in top_kcs_names
                            if kc in kc_to_idx])
        top_kcs_filtered = [listOfKC[i] for i in top_kc_idx]

        n_kc        = Q_mat.shape[1]
        n_kc_top    = len(top_kc_idx) 
        n_users     = df["user_id"].nunique()
        n_pairs     = n_users * n_kc_top  

        user_ids_list = sorted(df["user_id"].unique())
        user_to_idx   = {u: i for i, u in enumerate(user_ids_list)}
        X = {
            "users_kc": sparse.csr_matrix(np.empty((0, n_pairs))),
            "skills":   sparse.csr_matrix(np.empty((0, n_kc))),
            "attempts": sparse.csr_matrix(np.empty((0, n_kc * n_tw))),
            "wins":     sparse.csr_matrix(np.empty((0, n_kc * n_tw))),
            "fails":    sparse.csr_matrix(np.empty((0, n_kc))),
            "df":       np.empty((0, 5))
        }

        q = defaultdict(self.make_queue)

        for idx, stud_id in enumerate(df["user_id"].unique()):
            print(f"Stud_id: {idx+1}/{n_users}")

            df_stud = df[df["user_id"] == stud_id][
                ["user_id", "item_id", "timestamp", "correct", "inter_id"]
            ].sort_values("timestamp").to_numpy()

            X["df"] = np.vstack((X["df"], df_stud))
            user_idx = user_to_idx[stud_id]
            n_inter  = df_stud.shape[0]

            # Skills
            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"],
                                        sparse.csr_matrix(skills_temp)])

            # Attempts + Wins
            attempts = np.zeros((n_inter, n_kc * n_tw))
            wins     = np.zeros((n_inter, n_kc * n_tw))

            users_kc_local = np.zeros((n_inter, n_kc_top))

            for l, (item_id, t, correct) in enumerate(zip(
                    df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):

                for kc in dict_q_mat[item_id]:
                    attempts[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc].get_counters(t))
                    )
                    wins[l, kc*n_tw:(kc+1)*n_tw] = np.log(
                        1 + np.array(q[stud_id, kc, "correct"].get_counters(t))
                    )

                    if kc in top_kc_idx:
                        local_idx = top_kc_idx.index(kc)
                        users_kc_local[l, local_idx] = 1.0

                    q[stud_id, kc].push(t)
                    if correct:
                        q[stud_id, kc, "correct"].push(t)

            X["attempts"] = sparse.vstack([X["attempts"],
                                            sparse.csr_matrix(attempts)])
            X["wins"]     = sparse.vstack([X["wins"],
                                            sparse.csr_matrix(wins)])

            users_kc_row = sparse.hstack([
                sparse.csr_matrix((n_inter, user_idx * n_kc_top)),
                sparse.csr_matrix(users_kc_local),
                sparse.csr_matrix((n_inter, (n_users - user_idx - 1) * n_kc_top))
            ])
            X["users_kc"] = sparse.vstack([X["users_kc"], users_kc_row])

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

        # One-hot items
        enc_items = OneHotEncoder()
        X["items"] = enc_items.fit_transform(X["df"][:, 1].reshape(-1, 1))
        self.item_ids     = enc_items.categories_[0].tolist()
        self.user_ids     = user_ids_list
        self.top_kcs      = top_kcs_filtered  
        self.top_kc_idx   = top_kc_idx
        self.n_kc_top     = n_kc_top

        sparse_df = sparse.hstack([
            sparse.csr_matrix(X["df"]),
            X["users_kc"],
            X["items"],
            X["skills"],
            X["wins"],
            X["fails"],
            X["attempts"]
        ]).tocsr()

        return sparse_df, self.user_ids, self.item_ids, listOfKC
                

    def ComputeHistoryFeaturesRatio(self, Q_mat, df):

        dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)
        X = {
            "skills": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "ratio": sparse.csr_matrix(np.empty((0, Q_mat.shape[1] * self.n_tw))),
            "fails": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "df": np.empty((0, 5))
        }

        q = defaultdict(self.make_queue)

        for idx, stud_id in enumerate(df["user_id"].unique()):
            print("Stud_id:", idx+1, "/", len(df["user_id"].unique()))
            df_stud = df[df["user_id"] == stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]]
            df_stud = df_stud.sort_values("timestamp").to_numpy()

            X["df"] = np.vstack((X["df"], df_stud))

            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"], sparse.csr_matrix(skills_temp)])

            ratio = np.zeros((df_stud.shape[0], Q_mat.shape[1] * self.n_tw))
            for l, (item_id, t, correct) in enumerate(zip(df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                for kc in dict_q_mat[item_id]:
                    w = np.array(q[stud_id, kc, "correct"].get_counters(t))
                    a = np.array(q[stud_id, kc].get_counters(t))
                    
                    ratio[l, kc*self.n_tw:(kc+1)*self.n_tw] = np.log(
                        (1 + w) / (1 + a)
                    )
                    
                    q[stud_id, kc].push(t)
                    if correct:
                        q[stud_id, kc, "correct"].push(t)

            X["ratio"] = sparse.vstack([X["ratio"], sparse.csr_matrix(ratio)])

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

        enc_users = OneHotEncoder()
        enc_items = OneHotEncoder()
        X["users"] = enc_users.fit_transform(X["df"][:, 0].reshape(-1, 1))
        X["items"] = enc_items.fit_transform(X["df"][:, 1].reshape(-1, 1))
        self.user_ids = enc_users.categories_[0].tolist()
        self.item_ids = enc_items.categories_[0].tolist()

        listOfKC = []
        for kc_raw in df["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)
        sparse_df = sparse.hstack([
            sparse.csr_matrix(X["df"]),
            X["users"],
            X["items"],
            X["skills"],
            X["ratio"],  
            X["fails"],
        ]).tocsr()

        return sparse_df, self.user_ids, self.item_ids, listOfKC
    

    def ComputeHistoryfeaturesRatioAlpha(self,Q_mat,df):
        dict_q_mat = {i: set() for i in range(Q_mat.shape[0])}
        for item, kc in np.argwhere(Q_mat == 1):
            dict_q_mat[item].add(kc)

        n_kc    = Q_mat.shape[1]
        n_users = df["user_id"].nunique()
        n_pairs = n_users * n_kc  # nb colonnes pour alpha_{s,k}

        user_ids_list = sorted(df["user_id"].unique())
        user_to_idx   = {u: i for i, u in enumerate(user_ids_list)}
        X = {
            "users_kc": sparse.csr_matrix(np.empty((0, n_pairs))), 
            "skills": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "ratio": sparse.csr_matrix(np.empty((0, Q_mat.shape[1] * self.n_tw))),
            "fails": sparse.csr_matrix(np.empty((0, Q_mat.shape[1]))),
            "df": np.empty((0, 5))
        }

        q = defaultdict(self.make_queue)

        for idx, stud_id in enumerate(df["user_id"].unique()):
            print("Stud_id:", idx+1, "/", len(df["user_id"].unique()))
            df_stud = df[df["user_id"] == stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]]
            df_stud = df_stud.sort_values("timestamp").to_numpy()

            X["df"] = np.vstack((X["df"], df_stud))
            user_idx   = user_to_idx[stud_id]
            n_inter    = df_stud.shape[0]
            skills_temp = Q_mat[df_stud[:, 1].astype(int)]
            X["skills"] = sparse.vstack([X["skills"], sparse.csr_matrix(skills_temp)])

            ratio = np.zeros((df_stud.shape[0], Q_mat.shape[1] * self.n_tw))
            users_kc_local = np.zeros((n_inter, n_kc))
            for l, (item_id, t, correct) in enumerate(zip(df_stud[:, 1], df_stud[:, 2], df_stud[:, 3])):
                for kc in dict_q_mat[item_id]:
                    w = np.array(q[stud_id, kc, "correct"].get_counters(t))
                    a = np.array(q[stud_id, kc].get_counters(t))
                    
                    ratio[l, kc*self.n_tw:(kc+1)*self.n_tw] = np.log(
                        (1 + w) / (1 + a)
                    )
                    #user_kc
                    users_kc_local[l, kc] = 1.0
                    q[stud_id, kc].push(t)
                    if correct:
                        q[stud_id, kc, "correct"].push(t)

            X["ratio"] = sparse.vstack([X["ratio"], sparse.csr_matrix(ratio)])
            users_kc_row = sparse.hstack([
                sparse.csr_matrix((n_inter, user_idx * n_kc)),           # zéros avant
                sparse.csr_matrix(users_kc_local),                        # bloc local
                sparse.csr_matrix((n_inter, (n_users - user_idx - 1) * n_kc))  # zéros après
            ])
            X["users_kc"] = sparse.vstack([X["users_kc"], users_kc_row])

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

        
        enc_items = OneHotEncoder()
        X["items"] = enc_items.fit_transform(X["df"][:, 1].reshape(-1, 1))
        self.item_ids = enc_items.categories_[0].tolist()
        self.user_ids = user_ids_list
        listOfKC = []
        for kc_raw in df["KC"].unique():
            for elt in kc_raw.split("~~"):
                listOfKC.append(elt)
        listOfKC = np.unique(listOfKC)
        sparse_df = sparse.hstack([
            sparse.csr_matrix(X["df"]),
            X["users_kc"],  
            X["items"],
            X["skills"],
            X["ratio"],  
            X["fails"],
        ]).tocsr()

        return sparse_df, self.user_ids, self.item_ids, listOfKC