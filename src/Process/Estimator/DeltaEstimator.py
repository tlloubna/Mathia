import numpy as np 
from utils.this_queue import OurQueue
from scipy.optimize import minimize_scalar


class DeltaEstimator:
    def __init__(self, params:dict=None, prior_mean:float=0, 
                 prior_std=1, bounds:list[int]=[-3,3],
                 q_matrix=None):

        self.params = params
        self.intercept = params.get("intercept", 0)
        self.prior_mean = prior_mean      
        self.prior_std = prior_std
        self.bounds = bounds
        self.q_matrix = q_matrix
        
   
    def estimate_from_X(self, X_student, y_student):
        
        pipe = self.params["pipeline"]  
        scaler = pipe.named_steps["scaler"]
        lr = pipe.named_steps["lr"]
        coef = lr.coef_[0]
        intercept = lr.intercept_[0]
        X_scaled = scaler.transform(X_student)
        logits_base = X_scaled @ coef + intercept
       
        def neg_log_posterior(delta_j):
            logits = logits_base + delta_j
            p = 1 / (1 + np.exp(-logits))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = np.sum(y_student * np.log(p) + (1 - y_student) * np.log(1 - p))
            prior = -(delta_j - self.prior_mean)**2 / (2 * self.prior_std**2)
            return -(ll + prior)
        
        result = minimize_scalar(neg_log_posterior, bounds=self.bounds, method='bounded')
        return result.x
    


    
    
    









    