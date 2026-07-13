import numpy as np 
import json 
import src.Process.DAS3H as das3h
from collections import defaultdict
import re
class ZPD_KCS():
    def __init__(self,datajs=None,kclist=None,z1=0.2,z2=0.7,):
        self.datajs=datajs
        self.limit1=z1
        self.limit2=z2
        self.kcslist=kclist
        self.idx_to_json_id, self.json_id_to_idx=self.build_kc_mapping(kc_list=self.kcslist,data_js=self.datajs)
        self.children_of = defaultdict(list)
        for node in self.datajs:
            if node["parent"] != "0":
                self.children_of[node["parent"]].append(node["id"])
    def get_children_indices(self, kc_idx):
        """Depuis un indice DAS3H, récupérer les indices des enfants."""
        json_id = self.idx_to_json_id.get(kc_idx)
        if json_id is None:
            return []
        
        children_json_ids = self.children_of.get(json_id, [])
        children_indices = []
        for child_id in children_json_ids:
            if child_id in self.json_id_to_idx:
                children_indices.append(self.json_id_to_idx[child_id])
        return children_indices
    def build_kc_mapping(self,kc_list, data_js):
        def clean_name(name):
            return re.sub(r'\s*\[\d+\]$', '', name).strip()
        
        json_name_to_id = {node["name"]: node["id"] for node in data_js}
        idx_to_json_id = {}    # indice DAS3H → id JSON
        json_id_to_idx = {}    # id JSON → indice DAS3H
        not_found = []
        
        for idx, raw_name in enumerate(kc_list):
            name = clean_name(raw_name)
            if name in json_name_to_id:
                json_id = json_name_to_id[name]
                idx_to_json_id[idx] = json_id
                json_id_to_idx[json_id] = idx
            else:
                not_found.append((idx, raw_name))
        if not_found:
            print(f"{len(not_found)} KCs sans correspondance dans le JSON :")
            for idx, name in not_found[:10]:
                print(f"  idx={idx} : '{name}'")
        return idx_to_json_id, json_id_to_idx

    def ChooseItem(self, kc,  items_per_kc=None):
        items = items_per_kc.get(kc, [])
        if len(items) == 0:
            return None
        return np.random.choice(items)
   
    def find_best_child(self, kc_idx, params, queues, t_current, alpha_s, items_per_kc):
        children = self.get_children_indices(kc_idx)  
        
        if not children:
            return kc_idx
        
        for child_idx in children:
            items = items_per_kc.get(child_idx, [])
            if len(items)==0:
                continue
            pmr = self.computepmr(child_idx, items, params, queues, t_current, alpha_s)
            if pmr < self.limit1:
                deeper = self.find_best_child(child_idx, params, queues, t_current, alpha_s, items_per_kc)
                if deeper is not None:
                    return deeper
            elif pmr < self.limit2:
                return child_idx
        
        return kc_idx
    


    def computepmr(self,kc,items,params,queues,t_current,alpha_s):
        
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
        pmr = 1 / (1 + np.exp(-logit))
        return pmr

    def ChooseKC(self, t_current, student, queues, params, items_per_kc, kcs_introduced):
        alpha_s = params["alpha_s"][student]
        selected = {}
        all_pmrs = {}
        for kc_idx in kcs_introduced:
            items = items_per_kc.get(kc_idx, [])
            if len(items)<1:
                continue
            pmr = self.computepmr(kc_idx, items, params, queues, t_current, alpha_s)
            all_pmrs[kc_idx] = pmr
            if self.limit1 < pmr < self.limit2:
                selected[kc_idx] = pmr
        if not selected:
            if all_pmrs:
                best_kc = min(all_pmrs, key=lambda k: abs(all_pmrs[k] - self.limit1))
                return [best_kc]
            return []
        to_review = set()
        for kc_idx in selected:
            best = self.find_best_child(kc_idx, params, queues, t_current, alpha_s, items_per_kc)
            to_review.add(best)
        
        return list(to_review)
        
    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None, q_mat_=None,
                                student=None, queues=None, params=None,
                                t_current=None, items_per_kc=None, **kwargs):
        
        kcs = self.ChooseKC(t_current, student, queues, params, items_per_kc, kcs_introduced)
        if not kcs:
            return None, []
        kc = np.random.choice(kcs)  # choisir un KC parmi ceux de la ZPD
        item = self.ChooseItem(kc, items_per_kc)
        if item is None:
            return None, []
        return item, [kc]
        







        

            


        





    


