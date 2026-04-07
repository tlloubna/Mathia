
import random 
import numpy as np

class ThetaTresholdH():
    def __init__(self,theta_threshold=0.5,multi_kc=False):
        self.theta_threshold=theta_threshold
        self.multi_kc=multi_kc
    
    def ChooseItem(self, kc, q_mat_=None, items_per_kc=None):
        items = items_per_kc.get(kc, [])
        if len(items) == 0:
            return None
        return np.random.choice(items)
    def ChooseKC(self, t_current, kcs_introduced, student=None, queues=None, params=None, qmat_=None,items_per_kc=None):
        alpha_s = params["alpha_s"][student]
        best_kc = None
        best_dist = float("inf")
        
        for kc in kcs_introduced:
            items = items_per_kc.get(kc, [])
            if len(items) == 0:
                continue
            item = items[0]  # pas besoin de random ici, c'est juste pour évaluer
            
            delta_j = params["delta_j"][item]
            beta = params["beta_j"].get(kc, 0)
            h = 0.0
            if kc in queues:
                cw = queues[kc]["wins"].get_counters(t_current)
                ca = queues[kc]["attempts"].get_counters(t_current)
                for i in range(5):
                    h += params["theta_wins"][kc][i] * np.log(1 + cw[i])
                    h += params["theta_attempts"][kc][i] * np.log(1 + ca[i])
            
            logit = alpha_s - delta_j + beta + h
            pmr = 1 / (1 + np.exp(-logit))
            dist = abs(pmr - self.theta_threshold)
            
            if dist < best_dist:
                best_dist = dist
                best_kc = kc
        
        return best_kc
    

    def ChooseMultiKC(self, t_current, kcs_introduced, student=None, queues=None, params=None, qmat_=None, items_per_kc=None):
        alpha_s = params["alpha_s"][student]
        pmr_kc = {}
        for kc in kcs_introduced:
            items = items_per_kc.get(kc, [])
            if len(items) == 0:
                continue
            item = items[0]
            delta_j = params["delta_j"][item]
            beta = params["beta_j"].get(kc, 0)
            h = 0.0
            if kc in queues:
                cw = queues[kc]["wins"].get_counters(t_current)
                ca = queues[kc]["attempts"].get_counters(t_current)
                for i in range(5):
                    h += params["theta_wins"][kc][i] * np.log(1 + cw[i])
                    h += params["theta_attempts"][kc][i] * np.log(1 + ca[i])
            logit = alpha_s - delta_j + beta + h
            pmr_kc[kc] = 1 / (1 + np.exp(-logit))

        if not pmr_kc:
            return None, []

        kc_best = min(pmr_kc, key=lambda kc: abs(pmr_kc[kc] - self.theta_threshold))
        items_best = items_per_kc.get(kc_best, [])
        best_item = None
        best_score = float("inf")
        best_kcs = [kc_best]

        for item in items_best:
            kcs_item = [kc for kc in np.where(qmat_[item, :] == 1)[0]
                        if kc in pmr_kc]
            if len(kcs_item) == 0:
                continue
            score = np.mean([abs(pmr_kc[kc] - self.theta_threshold) for kc in kcs_item])
            if score < best_score:
                best_score = score
                best_item = item
                best_kcs = kcs_item

        return best_item, best_kcs
    

    def HeuristicTochooseItemfromQ(self,  week=None,kcs_introduced=None,q_mat_=None,student=None,queues=None,params=None,t_current=None,items_per_kc=None, **kwargs):

        if self.multi_kc:
            item,kcs=self.ChooseMultiKC(t_current, kcs_introduced, student, queues, params, q_mat_, items_per_kc)   
            if item is None:
                return None, []
            return item, kcs
        else: 
            kc = self.ChooseKC(t_current, kcs_introduced, student, queues, params,q_mat_, items_per_kc)
            if kc is None:
                return None, []
            item = self.ChooseItem(kc,q_mat_, items_per_kc)
            if item is None:
                return None, []
            return item, [kc]