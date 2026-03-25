
import numpy as np

class MuBackH():
    def __init__(self,mu:int=1,student=None,data=None,qmat=None,kc_list=None):
        self.mu = mu
        
        self.student=student
        self.data=data
        self.qmat=qmat
        self.kc_list=kc_list
    
    def ChooseKC(self, week, Kcs_introduced):
        target_week = max(0, week - self.mu)
        # Clamp à la taille de la liste introduite
        target_week = min(target_week, len(Kcs_introduced) - 1)
        return Kcs_introduced[target_week]
            
    def ChooseItem(self,kc):
        items = np.where(self.qmat[:, self.kc_list.index(kc)] == 1)[0]
        if len(items) == 0:
            return None
        return np.random.choice(items)
    
    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None):
        kc = self.ChooseKC(week, kcs_introduced)
        item = self.ChooseItem(kc)
        return item, [kc]
    

    

    