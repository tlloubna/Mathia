import numpy as np 
from utils.this_queue import OurQueue
from scipy.optimize import minimize_scalar
class alphaEstimator:
    def __init__(self, params:dict=None, prior_mean:float=0, 
                 prior_std=1, bounds:list[int]=[-3,3],
                 kmeans=None, scaler_kmeans=None, cluster_stats=None,q_matrix=None):

        self.params = params
        self.intercept = params.get("intercept", 0)
        self.prior_mean = prior_mean      
        self.prior_std = prior_std
        self.bounds = bounds
        self.kmeans = kmeans               
        self.scaler_kmeans = scaler_kmeans 
        self.cluster_stats = cluster_stats 
        self.q_matrix = q_matrix
        
    def _get_prior_from_cluster(self, X_student, y_student):
        if self.kmeans is None or self.cluster_stats is None:
            return self.prior_mean, self.prior_std
        n_interactions = X_student.shape[0]
        mean_success_rate = np.mean(y_student)
        item_ids = X_student[:, 1].toarray().flatten().astype(int)
        kcs_seen = set()
        for item in item_ids:
            kcs = np.where(self.q_matrix[item] == 1)[0]
            kcs_seen.update(kcs)
        n_kcs_seen = len(kcs_seen)
        deltas = [self.params["delta_j"][item] 
                for item in np.unique(item_ids) 
                if item in self.params["delta_j"]]
        mean_delta = np.mean(deltas) if len(deltas) > 0 else 0
        """features = np.array([[
            0,                  
            mean_success_rate,
            n_interactions,
            n_kcs_seen,
            mean_delta,
        ]])"""
        features = np.array([[0,mean_success_rate,mean_delta, ]])
        features_scaled = self.scaler_kmeans.transform(features)
        cluster = self.kmeans.predict(features_scaled)[0]
        
        prior_mean = self.cluster_stats[cluster]["mean_alpha"]
        prior_std = self.cluster_stats[cluster]["std_alpha"]
        n = X_student.shape[0]
        prior_std = prior_std * (1 + n / 50)
        
        return prior_mean, prior_std
    def estimate_from_X(self, X_student, y_student):
        
        pipe = self.params["pipeline"]  
        scaler = pipe.named_steps["scaler"]
        lr = pipe.named_steps["lr"]
        coef = lr.coef_[0]
        intercept = lr.intercept_[0]
        X_scaled = scaler.transform(X_student)
        logits_base = X_scaled @ coef + intercept
        prior_mean, prior_std = self._get_prior_from_cluster(X_student, y_student)

        def neg_log_posterior(alpha_s):
            logits = logits_base + alpha_s
            p = 1 / (1 + np.exp(-logits))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = np.sum(y_student * np.log(p) + (1 - y_student) * np.log(1 - p))
            prior = -(alpha_s - prior_mean)**2 / (2 * prior_std**2)
            return -(ll + prior)
        result = minimize_scalar(neg_log_posterior, bounds=self.bounds, method='bounded')
        return result.x
    


class AlphaEstimatorUsingClass:
    def __init__(self, params:dict=None, 
                 kmeans=None, scaler_kmeans=None, cluster_stats=None):

        self.params = params
        self.kmeans = kmeans               
        self.scaler_kmeans = scaler_kmeans 
        self.cluster_stats = cluster_stats 

    def _get_prior_from_cluster(self, X_student, y_student):
        mean_success_rate = np.mean(y_student)
        item_ids = X_student[:, 1].toarray().flatten().astype (int)
        deltas = [self.params["delta_j"][item] 
                for item in np.unique(item_ids) 
                if item in self.params["delta_j"]]
        mean_delta = np.mean(deltas) if len(deltas) > 0 else 0
        features = np.array([[mean_success_rate,mean_delta, ]])
        features_scaled = self.scaler_kmeans.transform(features)
        cluster = self.kmeans.predict(features_scaled)[0]
        mean_alpha = self.cluster_stats[cluster]["mean_alpha"]
        std_alpha = self.cluster_stats[cluster]["std_alpha"]
        return mean_alpha,std_alpha

    

    
    
    









    