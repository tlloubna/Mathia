# src/Process/Heuristics/ZPD_window.py
import numpy as np


class ZPD_window():
    """
    ZPD "pure" (Vygotsky) ne réviser que les compétences dans la zone proximale de
    développement, c.-à-d. dont le PMR est dans ]z1, z2[ — ni trop faciles
    (déjà acquises, PMR >= z2), ni trop difficiles (hors de portée, PMR <= z1).
    """
    def __init__(self, z1=0.2, z2=0.7):
        self.limit1 = z1
        self.limit2 = z2
    def computepmr(self, kc, items, params, queues, t_current, alpha_s):
        item = items[0]
        delta_j = params["delta_j"][int(item)]
        beta = params["beta_j"].get(kc, 0)
        h = 0.0
        if kc in queues:
            cw = queues[kc]["wins"].get_counters(t_current)
            ca = queues[kc]["attempts"].get_counters(t_current)
            for i in range(5):
                h += params["theta_wins"][kc][i] * np.log(1 + cw[i])
                h += params["theta_attempts"][kc][i] * np.log(1 + ca[i])
        logit = alpha_s - delta_j + beta + h
        return 1 / (1 + np.exp(-logit))

    def ChooseKC(self, t_current, student, queues, params, items_per_kc,
                 kcs_introduced):
        alpha_s = params["alpha_s"][student]
        in_zpd = {}        
        all_pmrs = {}
        for kc in kcs_introduced:
            items = items_per_kc.get(kc, [])
            if len(items) < 1:
                continue
            pmr = self.computepmr(kc, items, params, queues, t_current, alpha_s)
            all_pmrs[kc] = pmr
            if self.limit1 < pmr < self.limit2:
                in_zpd[kc] = pmr
        if in_zpd:
            return list(in_zpd.keys())
        if all_pmrs:
            best = min(all_pmrs, key=lambda k: abs(all_pmrs[k] - self.limit1))
            return [best]
        return []

    def ChooseItem(self, kc, items_per_kc):
        items = items_per_kc.get(kc, [])
        if len(items) == 0:
            return None
        return int(np.random.choice(items))

    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None,
                                   q_mat_=None, student=None, queues=None,
                                   params=None, t_current=None,
                                   items_per_kc=None, **kwargs):
        kcs = self.ChooseKC(t_current, student, queues, params,
                            items_per_kc, kcs_introduced)
        if not kcs:
            return None, []
        kc = int(np.random.choice(kcs))  
        item = self.ChooseItem(kc, items_per_kc)
        if item is None:
            return None, []
        return item, [kc]