

import numpy as np
import pandas as pd
import src.Process.DAS3H as das3H


def ComputePMR(model:das3H.DAS3HModel,student,kc,t,q,item):
        params=model.get_params()
        alpha=params["alpha_s"].get(student, 0.0)
        delta=params["delta_j"].get(item, 0.0) if item is not None else 0.0
        beta=params["beta_k"].get(kc, 0.0)
        theta_wins=params["theta_wins"].get(kc, 0.0)
        theta_attempts=params["theta_attempts"].get(kc, 0.0)
        

        wins_count = q[(kc,"wins")].get_counters(t)  
        attempts_count = q[(kc,"attempts")].get_counters(t) 
        h=0.0
        for w in range(len(theta_wins)):
            log_wins=np.log(1+wins_count[w])
            log_att=np.log(1+attempts_count[w])
            h+=theta_wins[w]*log_wins+theta_attempts[w]*log_att
        #Total fails per student and kc
        
        logit=alpha+delta+beta+h
        
        return 1/(1+np.exp(-logit))