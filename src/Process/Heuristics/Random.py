
import numpy as np
class RandomH:
    def __init__(self, qmat=None, kc_list=None):

        self.qmat = qmat
        self.kc_list = kc_list
        

    def select_kc(self):
        return np.random.choice(self.kc_list)
    
    def select_item(self, kc):
        items = np.where(self.qmat[:, self.kc_list.index(kc)] == 1)[0]
        if len(items) == 0:
            return None
        return np.random.choice(items)
    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None):
        if kcs_introduced and len(kcs_introduced) > 0:
            kc = np.random.choice(kcs_introduced)  # déjà correct car choice sur la liste
        else:
            return None, []
        item = self.select_item(kc)
        if item is None:
            return None, []
        return item, [kc]