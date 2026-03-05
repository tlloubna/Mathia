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

    def fit(self, X, user_ids, item_ids, kc_list, n_tw=5, perc_init=0.2):
        
        self.n_users  = len(user_ids)
        self.n_items  = len(item_ids)
        self.n_kc     = len(kc_list)
        self.n_tw     = n_tw
        self.user_ids = list(user_ids)
        self.item_ids = list(item_ids)
        self.kc_list  = list(kc_list)
        y = X[:, 3].toarray().flatten()
        timestamps = X[:, 2].toarray().flatten()
        users_col = X[:, 0].toarray().flatten()
        
        train_indices = []
        test_indices = []
        
        for user in np.unique(users_col):
            user_mask = np.where(users_col == user)[0]
            user_sorted = user_mask[np.argsort(timestamps[user_mask])]
            split = max(1, round(perc_init * len(user_sorted)))
            train_indices.extend(user_sorted[:split].tolist())
            test_indices.extend(user_sorted[split:].tolist())
        
        cols = list(range(X.shape[1]))
        cols.remove(3)
        X_features = X[:, cols]
        
        X_train = X_features[train_indices]
        X_test  = X_features[test_indices]
        y_train = y[train_indices]
        y_test  = y[test_indices]
        pipe = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("lr", LogisticRegression(solver="saga", max_iter=500, C=self.C))
        ])

        pipe.fit(X_train, y_train)
        self.model = pipe

        y_pred = pipe.predict_proba(X_test)[:, 1]

        return {
            "AUC":    roc_auc_score(y_test, y_pred),
            "NLL":    log_loss(y_test, y_pred),
            "RMSE":   np.sqrt(mean_squared_error(y_test, y_pred)),
            "y_test": y_test,
            "y_pred": y_pred,
            "FPR":    roc_curve(y_test, y_pred)[0],
            "TPR":    roc_curve(y_test, y_pred)[1],
        }
    

    def get_params(self):
    
        lr   = self.model.named_steps["lr"]
        coef = lr.coef_[0]
        
        nu  = self.n_users
        ni  = self.n_items
        nk  = self.n_kc    
        ntw = self.n_tw
        
        # Structure fixe et connue
        offset = 4  # colonnes df avant OHE
        
        i_u = (offset,           offset + nu)
        i_i = (i_u[1],           i_u[1] + ni)
        i_k = (i_i[1],           i_i[1] + nk)
        i_w = (i_k[1],           i_k[1] + nk * ntw)
        i_f = (i_w[1],           i_w[1] + nk)
        i_a = (i_f[1],           i_f[1] + nk * ntw)
        
        
        assert i_a[1] == len(coef), (
            f"Structure incohérente : attendu {i_a[1]} coefs, "
            f"obtenu {len(coef)}. "
            f"Vérifie n_kc={nk}, n_tw={ntw}, n_users={nu}, n_items={ni}"
        )
        
        
        return {
            "intercept"     : float(lr.intercept_[0]),
            "alpha_s"       : dict(zip(self.user_ids, coef[i_u[0]:i_u[1]])),
            "delta_j"       : dict(zip(self.item_ids, -coef[i_i[0]:i_i[1]])),
            "beta_k"        : dict(zip(self.kc_list,  coef[i_k[0]:i_k[1]])),
            "theta_wins"    : {kc: coef[i_w[0]:i_w[1]].reshape(nk, ntw)[i]
                            for i, kc in enumerate(self.kc_list)},
            "theta_fails"   : dict(zip(self.kc_list,  coef[i_f[0]:i_f[1]])),
            "theta_attempts": {kc: coef[i_a[0]:i_a[1]].reshape(nk, ntw)[i]
                            for i, kc in enumerate(self.kc_list)},
                            }
    
    def get_paramAlphask(self):
        lr   = self.model.named_steps["lr"]
        coef = lr.coef_[0]
        nu  = self.n_users
        ni  = self.n_items
        nk  = self.n_kc
        ntw = self.n_tw

        offset = 4  # colonnes df avant OHE
        i_usk = (offset,          offset + nu * nk)  # alpha_{s,k}
        i_i   = (i_usk[1],        i_usk[1] + ni)
        i_k   = (i_i[1],          i_i[1] + nk)
        i_w   = (i_k[1],          i_k[1] + nk * ntw)
        i_f   = (i_w[1],          i_w[1] + nk)
        i_a   = (i_f[1],          i_f[1] + nk * ntw)

        assert i_a[1] == len(coef), (
            f"Structure incohérente : attendu {i_a[1]} coefs, "
            f"obtenu {len(coef)}. "
            f"Vérifie n_kc={nk}, n_tw={ntw}, n_users={nu}, n_items={ni}"
        )

        # Construire alpha_sk — dict[(user_id, kc)] = coef
        alpha_sk_coefs = coef[i_usk[0]:i_usk[1]].reshape(nu, nk)
        alpha_sk = {}
        for i, user_id in enumerate(self.user_ids):
            for j, kc in enumerate(self.kc_list):
                alpha_sk[(user_id, kc)] = float(alpha_sk_coefs[i, j])

        return {
            "intercept"     : float(lr.intercept_[0]),
            "alpha_sk"      : alpha_sk,   
            "delta_j"       : dict(zip(self.item_ids, -coef[i_i[0]:i_i[1]])),
            "beta_k"        : dict(zip(self.kc_list,   coef[i_k[0]:i_k[1]])),
            "theta_wins"    : {kc: coef[i_w[0]:i_w[1]].reshape(nk, ntw)[i]
                            for i, kc in enumerate(self.kc_list)},
            "theta_fails"   : dict(zip(self.kc_list,   coef[i_f[0]:i_f[1]])),
            "theta_attempts": {kc: coef[i_a[0]:i_a[1]].reshape(nk, ntw)[i]
                            for i, kc in enumerate(self.kc_list)},
        }

    def get_paramsRatio(self):

        lr   = self.model.named_steps["lr"]
        coef = lr.coef_[0]
        
        nu  = self.n_users
        ni  = self.n_items
        nk  = self.n_kc    
        ntw = self.n_tw
        
        offset = 4 
        
        i_u = (offset,   offset + nu)
        i_i = (i_u[1],   i_u[1] + ni)
        i_k = (i_i[1],   i_i[1] + nk)
        i_r = (i_k[1],   i_k[1] + nk * ntw)   
        i_f = (i_r[1],   i_r[1] + nk)          

        assert i_f[1] == len(coef), (
            f"Structure incohérente : attendu {i_f[1]} coefs, "
            f"obtenu {len(coef)}. "
            f"Vérifie n_kc={nk}, n_tw={ntw}, n_users={nu}, n_items={ni}"
        )
        
        return {
            "intercept"    : float(lr.intercept_[0]),
            "alpha_s"      : dict(zip(self.user_ids, coef[i_u[0]:i_u[1]])),
            "delta_j"      : dict(zip(self.item_ids, -coef[i_i[0]:i_i[1]])),
            "beta_k"       : dict(zip(self.kc_list,  coef[i_k[0]:i_k[1]])),
            "theta_ratio"  : {kc: coef[i_r[0]:i_r[1]].reshape(nk, ntw)[i]   
                            for i, kc in enumerate(self.kc_list)},
            "theta_fails"  : dict(zip(self.kc_list,  coef[i_f[0]:i_f[1]])),
        }

    def get_params_alphasbestk(self,nk_top,top_kcs):

        lr   = self.model.named_steps["lr"]
        coef = lr.coef_[0]

        nu     = self.n_users
        ni     = self.n_items
        nk     = self.n_kc
        ntw    = self.n_tw

        offset = 4

        # users_kc — seulement n_kc_top KCs
        i_usk = (offset,       offset + nu * nk_top)
        i_i   = (i_usk[1],     i_usk[1] + ni)
        i_k   = (i_i[1],       i_i[1] + nk)
        i_w   = (i_k[1],       i_k[1] + nk * ntw)
        i_f   = (i_w[1],       i_w[1] + nk)
        i_a   = (i_f[1],       i_f[1] + nk * ntw)

        assert i_a[1] == len(coef), (
            f"Structure incohérente : attendu {i_a[1]}, "
            f"obtenu {len(coef)}"
        )

        
        alpha_sk_coefs = coef[i_usk[0]:i_usk[1]].reshape(nu, nk_top)
        alpha_sk = {}
        for i, user_id in enumerate(self.user_ids):
            for j, kc in enumerate(top_kcs):  # ← top_kcs seulement
                alpha_sk[(user_id, kc)] = float(alpha_sk_coefs[i, j])

        return {
            "intercept"     : float(lr.intercept_[0]),
            "alpha_sk"      : alpha_sk,
            "delta_j"       : dict(zip(self.item_ids, -coef[i_i[0]:i_i[1]])),
            "beta_k"        : dict(zip(self.kc_list,   coef[i_k[0]:i_k[1]])),
            "theta_wins"    : {kc: coef[i_w[0]:i_w[1]].reshape(nk, ntw)[i]
                            for i, kc in enumerate(self.kc_list)},
            "theta_fails"   : dict(zip(self.kc_list,   coef[i_f[0]:i_f[1]])),
            "theta_attempts": {kc: coef[i_a[0]:i_a[1]].reshape(nk, ntw)[i]
                            for i, kc in enumerate(self.kc_list)},
        }
    
    #Trouver les params pour la nouvelle fonction HistoryfeaturesratioAlpha
    def get_params_AlphaRatio(self):
        lr   = self.model.named_steps["lr"]
        coef = lr.coef_[0]
        nu  = self.n_users
        ni  = self.n_items
        nk  = self.n_kc
        ntw = self.n_tw

        offset = 4  # colonnes df avant OHE
        i_usk = (offset,          offset + nu * nk)  # alpha_{s,k}
        i_i   = (i_usk[1],        i_usk[1] + ni)
        i_k   = (i_i[1],          i_i[1] + nk)
        i_r = (i_k[1],   i_k[1] + nk * ntw)   
        i_f = (i_r[1],   i_r[1] + nk)
        assert i_f[1] == len(coef), (
            f"Structure incohérente : attendu {i_f[1]} coefs, "
            f"obtenu {len(coef)}. "
            f"Vérifie n_kc={nk}, n_tw={ntw}, n_users={nu}, n_items={ni}"
        )
        alpha_sk_coefs = coef[i_usk[0]:i_usk[1]].reshape(nu, nk)
        alpha_sk = {}
        for i, user_id in enumerate(self.user_ids):
            for j, kc in enumerate(self.kc_list):
                alpha_sk[(user_id, kc)] = float(alpha_sk_coefs[i, j])
        


        return {
            "intercept"     : float(lr.intercept_[0]),
            "alpha_sk"      : alpha_sk,   
            "delta_j"       : dict(zip(self.item_ids, -coef[i_i[0]:i_i[1]])),
            "beta_k"        : dict(zip(self.kc_list,   coef[i_k[0]:i_k[1]])),
            "theta_ratio"  : {kc: coef[i_r[0]:i_r[1]].reshape(nk, ntw)[i]   
                            for i, kc in enumerate(self.kc_list)},
            "theta_fails"   : dict(zip(self.kc_list,   coef[i_f[0]:i_f[1]])),
            
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

    def predict_signle_Alphask(self, user_id, item_id: int, kc_list: list, history: dict) -> float:
        
        p = self.get_paramAlphask()

        alpha = sum(p["alpha_sk"].get((user_id, kc), 0.0) for kc in kc_list)
        delta = p["delta_j"].get(item_id, 0.0)
        beta  = sum(p["beta_k"].get(kc, 0.0) for kc in kc_list)

        h_theta = 0.0
        for kc in kc_list:
            h_theta += np.dot(p["theta_wins"][kc],     history[kc]["wins"])
            h_theta += np.dot(p["theta_attempts"][kc], history[kc]["attempts"])
            h_theta += p["theta_fails"][kc]            * history[kc]["fails"]

        logit = alpha - delta + beta + h_theta + p["intercept"]
        return float(1.0 / (1.0 + np.exp(-logit)))
    

    def predict_singleRatio(self, user_id, item_id: int, kc_list: list, history: dict) -> float:
        p = self.get_paramsRatio()
        alpha = p["alpha_s"].get(user_id, 0.0)
        delta = p["delta_j"].get(item_id, 0.0)
        beta  = sum(p["beta_k"].get(kc, 0.0) for kc in kc_list)
        h_theta = 0.0
        for kc in kc_list:
            h_theta += np.dot(p["theta_ratio"][kc], history[kc]["ratio"])  
            h_theta += p["theta_fails"][kc]         * history[kc]["fails"]

        logit = alpha - delta + beta + h_theta + p["intercept"]
        return float(1.0 / (1.0 + np.exp(-logit)))
    
    def predict_SingleRatioAlpha(self, user_id, item_id: int, kc_list: list, history: dict) -> float:
        p=self.get_params_AlphaRatio()

        alpha = sum(p["alpha_sk"].get((user_id, kc), 0.0) for kc in kc_list)
        delta = p["delta_j"].get(item_id, 0.0)
        beta  = sum(p["beta_k"].get(kc, 0.0) for kc in kc_list)
        h_theta = 0.0
        for kc in kc_list:
            h_theta += np.dot(p["theta_ratio"][kc], history[kc]["ratio"])  
            h_theta += p["theta_fails"][kc]  * history[kc]["fails"]
            
        logit = alpha - delta + beta + h_theta + p["intercept"]
        return float(1.0 / (1.0 + np.exp(-logit)))
    
    def predict_proba(self, X_new) -> np.ndarray:
        """
        X_new : matrice sparse CSR (sans la colonne 'correct')
        """
        return self.model.predict_proba(X_new)[:, 1]