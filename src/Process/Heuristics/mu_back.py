
import numpy as np

class MuBackH():
    def __init__(self, mu=1, kc_list=None):
        self.mu = mu
    
        self.kc_list = list(kc_list)
        self.last_review = {}  # {kc: dernière semaine de révision}

    def reset(self):
        """Appeler au début de chaque nouvel étudiant"""
        self.last_review = {}

    def update(self, kc, week_num):
        """Appeler après chaque exo dans Loopweek"""
        self.last_review[kc] = week_num

    def ChooseKC(self, week_num, kcs_introduced):
        #on cherche le kc dont le gap est plus proche de mu : c pour ça 
        #on fait numero de la semaine - la dernière semaine de revision du KC 
        #on selectionnne ensuite le KC avec le gap le plus proche de mu
        never_reviewed = [kc for kc in kcs_introduced if kc not in self.last_review]
        if never_reviewed:
            return never_reviewed[np.random.randint(len(never_reviewed))]
        best_kc = None
        best_diff = float("inf")
        
        for kc in kcs_introduced:
            gap = week_num - self.last_review[kc]  
            diff = abs(gap - self.mu)
            if diff < best_diff:
                best_diff = diff
                best_kc = kc
        
        return best_kc

   
    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None, q_mat_=None, items_per_kc=None, **kwargs):
        kc = self.ChooseKC(week, kcs_introduced)
        if kc is None:
            return None, []
        items = items_per_kc.get(kc, [])
        if len(items) == 0:
            return None, []
        item = np.random.choice(items)
        return item, [kc]