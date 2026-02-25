import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


class DAS3HModel:

    def __init__(self, C=1.0):
        self.C = C
        self.model = None
        self.scaler = None

        # --- Metadonnees stockees au moment du fit ---
        self.n_users   = None
        self.n_items   = None
        self.n_kc      = None
        self.n_tw      = None
        self.user_ids  = None   # liste ordonnee des user_id (ordre OneHotEncoder)
        self.item_ids  = None   # liste ordonnee des item_id
        self.kc_list   = None   # liste ordonnee des noms de KC

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, X, user_ids: list, item_ids: list, kc_list: list, n_tw: int = 5):
        
        # Sauvegarder les metadonnees
        self.n_users  = len(user_ids)
        self.n_items  = len(item_ids)
        self.n_kc     = len(kc_list)
        self.n_tw     = n_tw
        self.user_ids = list(user_ids)
        self.item_ids = list(item_ids)
        self.kc_list  = list(kc_list)

        # y = colonne 3 (correct)
        y = X[:, 3].toarray().flatten()

        # X_features = toutes les colonnes sauf la 3
        cols = list(range(X.shape[1]))
        cols.remove(3)
        X_features = X[:, cols]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )

        # Pipeline
        pipe = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("lr", LogisticRegression(solver="saga", max_iter=500, C=self.C))
        ])

        pipe.fit(X_train, y_train)
        self.model = pipe

        y_pred = pipe.predict_proba(X_test)[:, 1]

        return {
            "AUC":    roc_auc_score(y_test, y_pred),
            "ACC":    accuracy_score(y_test, np.round(y_pred)),
            "NLL":    log_loss(y_test, y_pred),
            "RMSE":   np.sqrt(mean_squared_error(y_test, y_pred)),
            "y_test": y_test,
            "y_pred": y_pred,
            "FPR":    roc_curve(y_test, y_pred)[0],
            "TPR":    roc_curve(y_test, y_pred)[1],
        }

    def get_params(self) -> dict:
        
        if self.model is None:
            raise RuntimeError("Le modele n'a pas encore ete entraine (appelle fit() d'abord).")

        lr   = self.model.named_steps["lr"]
        coef = lr.coef_[0]

        nu  = self.n_users
        ni  = self.n_items
        ntw = self.n_tw
        offset = 4   
        reste_apres_users_items = len(coef) - offset - nu - ni
        nk_reel = reste_apres_users_items // (2 + 2 * ntw)

        
        attendu = nk_reel * (2 + 2 * ntw)
        if attendu != reste_apres_users_items:
            raise ValueError(
                f"Impossible de déduire nk_reel : reste={reste_apres_users_items}, "
                f"nk_reel={nk_reel}, attendu={attendu}. "
                f"Vérifiez n_tw={ntw} et la structure de sparse_df."
            )

        if nk_reel != self.n_kc:
            print(f"[WARNING] n_kc déclaré={self.n_kc} mais n_kc dans coef_={nk_reel} "
                  f"({self.n_kc - nk_reel} KC absents du train set → ignorés)")

        #
        i_u_start = offset
        i_u_end   = i_u_start + nu

        i_i_start = i_u_end
        i_i_end   = i_i_start + ni

        i_k_start = i_i_end
        i_k_end   = i_k_start + nk_reel

        i_w_start = i_k_end
        i_w_end   = i_w_start + nk_reel * ntw

        i_f_start = i_w_end
        i_f_end   = i_f_start + nk_reel

        i_a_start = i_f_end
        i_a_end   = i_a_start + nk_reel * ntw
        alpha_s_arr        = coef[i_u_start : i_u_end]
        delta_j_arr        = -coef[i_i_start : i_i_end]   # signe inverse = difficulte
        beta_k_arr         = coef[i_k_start : i_k_end]
        theta_wins_arr     = coef[i_w_start : i_w_end].reshape(nk_reel, ntw)
        theta_fails_arr    = coef[i_f_start : i_f_end]
        theta_attempts_arr = coef[i_a_start : i_a_end].reshape(nk_reel, ntw)

        kc_reel = self.kc_list[:nk_reel]

        return {
            "intercept"     : float(lr.intercept_[0]),
            "alpha_s"       : dict(zip(self.user_ids, alpha_s_arr.tolist())),
            "delta_j"       : dict(zip(self.item_ids, delta_j_arr.tolist())),
            "beta_k"        : dict(zip(kc_reel,       beta_k_arr.tolist())),
            "theta_wins"    : {kc: theta_wins_arr[i]     for i, kc in enumerate(kc_reel)},
            "theta_fails"   : dict(zip(kc_reel, theta_fails_arr.tolist())),
            "theta_attempts": {kc: theta_attempts_arr[i] for i, kc in enumerate(kc_reel)},
            # Infos utiles
            "_nk_declared"  : self.n_kc,
            "_nk_reel"      : nk_reel,
            "_kc_missing"   : self.kc_list[nk_reel:],   # KC absents du train set
        }
    
    def predict_single(self, user_id, item_id: int, kc_list: list, history: dict) -> float:
        
        p = self.get_params()

        alpha = p["alpha_s"].get(user_id, 0.0)
        delta = p["delta_j"].get(item_id, 0.0)
        beta  = sum(p["beta_k"].get(kc, 0.0) for kc in kc_list)

        h_theta = 0.0
        for kc in kc_list:
            h_theta += np.dot(p["theta_wins"][kc],     history[kc]["wins"])
            h_theta += np.dot(p["theta_attempts"][kc], history[kc]["attempts"])
            h_theta += p["theta_fails"][kc]            * history[kc]["fails"]

        logit = alpha - delta + beta + h_theta + p["intercept"]
        return float(1.0 / (1.0 + np.exp(-logit)))

    # ------------------------------------------------------------------
    # PREDICTION BATCH (sklearn)
    # ------------------------------------------------------------------
    def predict_proba(self, X_new) -> np.ndarray:
        """
        X_new : matrice sparse CSR (sans la colonne 'correct')
        """
        return self.model.predict_proba(X_new)[:, 1]