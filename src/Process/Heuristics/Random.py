
import numpy as np
class RandomH:
    def __init__(self,  kc_list=None):


        self.kc_list = kc_list
        

    def select_kc(self):
        return np.random.choice(self.kc_list)
    
   
    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None,q_mat_=None,items_per_kc=None, **kwargs):
        if kcs_introduced and len(kcs_introduced) > 0:
            kc = np.random.choice(kcs_introduced)  # déjà correct car choice sur la liste
        else:
            return None, []
        items = items_per_kc.get(kc, [])
        if len(items) == 0:
            return None, []
        item = np.random.choice(items)
        return item, [kc]