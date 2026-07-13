
import numpy as np
from src.Process.Heuristics.ZPD_KCs import ZPD_KCS

class ZPD_propagation(ZPD_KCS):
    """
    pmr_parent_eff = (1 - lam) * pmr_parent_brut+ lam * moyenne_pondérée(pmr_enfants_eff)
    lam=0 redonne exactement ZPD_KCS. 
    """
    def __init__(self, datajs=None, kclist=None, z1=0.2, z2=0.7,
                 lam=0.5, weights=None):
        super().__init__(datajs=datajs, kclist=kclist, z1=z1, z2=z2)
        self.lam = lam
        self.weights = weights or {}     
        self._pmr_cache = None          
    def _propagate(self, pmr_raw):
        pmr_eff = dict(pmr_raw)
        def resolve(idx):
            children = self.get_children_indices(idx)
            if not children:
                return pmr_eff[idx]             
            contribs, ws = [], []
            for c in children:
                if c not in pmr_eff:
                    continue
                child_val = resolve(c)            
                w = self.weights.get(c, 1.0)
                contribs.append(w * child_val)
                ws.append(w)
            if ws:
                contribution = sum(contribs) / sum(ws)
                pmr_eff[idx] = ((1 - self.lam) * pmr_raw[idx] + self.lam * contribution)
            return pmr_eff[idx]
        for idx in list(pmr_raw.keys()):
            resolve(idx)
        return pmr_eff

    def _build_cache(self, params, queues, t_current, alpha_s, items_per_kc,
                     kcs_introduced):
        pmr_raw = {}
        for kc in kcs_introduced:
            items = items_per_kc.get(kc, [])
            if len(items) < 1:
                continue
            pmr_raw[kc] = super().computepmr(kc, items, params, queues,
                                             t_current, alpha_s)
        self._pmr_cache = self._propagate(pmr_raw)

    def computepmr(self, kc, items, params, queues, t_current, alpha_s):
        if self._pmr_cache is not None and kc in self._pmr_cache:
            return self._pmr_cache[kc]
        return super().computepmr(kc, items, params, queues, t_current, alpha_s)

    def ChooseKC(self, t_current, student, queues, params, items_per_kc,
                 kcs_introduced):
        alpha_s = params["alpha_s"][student]
        self._build_cache(params, queues, t_current, alpha_s,
                          items_per_kc, kcs_introduced)
        result = super().ChooseKC(t_current, student, queues, params,
                                  items_per_kc, kcs_introduced)
        self._pmr_cache = None           
        return result