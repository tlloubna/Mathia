import src.Process.DAS3H as das3H
from utils.this_queue import OurQueue
import src.datamodel.Studentdata as STD
from collections import defaultdict
import src.Process.Simulation.utils as simu_utils
import random
import pandas as pd
import numpy as np
class SimulationH():
    def __init__(self,students=None,exos=None,kcs=None,data:pd.DataFrame=None,model:das3H.DAS3HModel=None,qmat=None,heuristic=None,history:bool=False,
                 weeks_to_simulate=16,T_max_review_min=60,t0=0):
        self.students=students
        self.exos=exos
        self.kcs=kcs
        self.data=data
        self.model:das3H.DAS3HModel=model
        self.qmat=qmat
        self.heuristic=heuristic
        self.seed=42
        self.history=history
        self.weeks_to_simulate=weeks_to_simulate
        self.T_max_review_min=T_max_review_min
        self.simulation_results=[]
        self.t0=t0
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.items_per_kc = {}
        for kc in range(qmat.shape[1]):
            items = np.where(qmat[:, kc] == 1)[0]
            if len(items) > 0:
                self.items_per_kc[kc] = items

    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def Loopweek_csTime(self, student, week_num, params, kcs_introduced, queues, t_start):
        t_current = t_start
        t_end = t_start + self.T_max_review_min * 60
        
        while t_current < t_end:
            item, kcs = self.heuristic.HeuristicTochooseItemfromQ(
                week=week_num, kcs_introduced=kcs_introduced,q_mat_=self.qmat,student=student,
                queues=queues,params=params,t_current=t_current, items_per_kc=self.items_per_kc
            )
            if item is None:
                break

            alpha_s = params["alpha_s"][student]
            delta_j = params["delta_j"][item]
            beta_sum = sum(params["beta_j"].get(kc, 0) for kc in kcs)
            h_wins = 0.0
            h_attempts = 0.0
            for kc in kcs:
                if kc not in queues:
                    queues[kc] = {"wins": OurQueue(), "attempts": OurQueue()}
                counters_w = queues[kc]["wins"].get_counters(t_current)
                counters_a = queues[kc]["attempts"].get_counters(t_current)
                for i in range(5):
                    h_wins += params["theta_wins"][kc][i] * np.log(1 + counters_w[i])
                    h_attempts += params["theta_attempts"][kc][i] * np.log(1 + counters_a[i])

            logit = alpha_s - delta_j + beta_sum + h_wins + h_attempts
            p_correct = self.sigmoid(logit)
            correct = int(np.random.random() < p_correct)

            # Push dans les queues
            for kc in kcs:
                queues[kc]["attempts"].push(t_current)
                if correct:
                    queues[kc]["wins"].push(t_current)
                if hasattr(self.heuristic, 'update'):
                    self.heuristic.update(kc, week_num)
            exo_duration = np.random.randint(2, 8) * 60
            t_current += exo_duration
            self.simulation_results.append({
                "student": student, "week": week_num,
                "item": item, "kcs": kcs,
                "p_correct": p_correct, "correct": correct,
                "timestamp": t_current,
            })
    

   
    
    def build_curriculum(self, kc_list, n_weeks=16):
        kc_list = list(kc_list)
        kcs_per_week = max(1, len(kc_list) // n_weeks)
        # Garder seulement kcs_per_week * n_weeks KCs
        kc_list = kc_list[:kcs_per_week * n_weeks]
        curriculum = {}
        for w in range(n_weeks):
            start = w * kcs_per_week
            curriculum[w] = kc_list[start:start + kcs_per_week]
        return curriculum
    
    def simulate(self, params):
        curriculum = self.build_curriculum(self.kcs, self.weeks_to_simulate)
        queues_all = {}
        weekly_pmr = {w: [] for w in range(self.weeks_to_simulate)}
        for student in self.students:
            queues = {}
            kcs_introduced = []
            if hasattr(self.heuristic, 'reset'):
                self.heuristic.reset()
            for week in range(self.weeks_to_simulate):
                new_kcs = curriculum.get(week, [])
                t_start = week * 7 * 24 * 3600
                for kc in new_kcs:
                    if kc not in queues:
                        queues[kc] = {"wins": OurQueue(), "attempts": OurQueue()}
                    queues[kc]["attempts"].push(week * 7 * 24 * 3600)
        
                kcs_introduced.extend(new_kcs)
                self.Loopweek_csTime(student, week, params, kcs_introduced, queues, t_start,)
                pmr_all=self.compute_pmr_all_kcs(student, params, queues, t_start)
                w_key = week + 1
                if w_key not in weekly_pmr:
                    weekly_pmr[w_key] = []
                weekly_pmr[w_key].append(pmr_all)
            queues_all[student] = queues
        results_pmr = {
            w: {"mean": np.mean(vals), "std": np.std(vals)}
            for w, vals in weekly_pmr.items() if vals
        }
        retention_pmr = self.evaluate_retention(params, queues_all, curriculum)
        global_pmr = self.compute_global_pmr(results_pmr, retention_pmr)
        return results_pmr, None, retention_pmr,global_pmr

    def evaluate_retention(self, params, queues_all, curriculum, retention_weeks=5):
        all_kcs = []
        for w in range(self.weeks_to_simulate):
            all_kcs.extend(curriculum.get(w, []))
        t_end_learning= self.t0 + self.weeks_to_simulate * 7 * 24 * 3600
        week_pmr={d: [] for d in range(retention_weeks+1)}
        for student, queues in queues_all.items():
            for week in range(retention_weeks+1):
                t_eval = t_end_learning + week * 7 * 24 * 3600
                pmr_per_kc = []
                for kc in all_kcs:
                    items_kc = self.items_per_kc.get(kc, [])
                    if len(items_kc) == 0:
                        delta_j=-1
                    else:
                        item = items_kc[0]
                        delta_j = params["delta_j"][item]
                    alpha_s = params["alpha_s"][student]
                    beta = params["beta_j"].get(kc, 0)
                    if kc in queues:
                        cw = queues[kc]["wins"].get_counters(t_eval)
                        ca = queues[kc]["attempts"].get_counters(t_eval)
                    else:
                        cw = [0] * 5
                        ca = [0] * 5

                    h = sum(
                        params["theta_wins"][kc][i] * np.log(1 + cw[i])
                        + params["theta_attempts"][kc][i] * np.log(1 + ca[i])
                        for i in range(5)
                    )
                    logit = alpha_s - delta_j + beta + h
                    pmr_per_kc.append(self.sigmoid(logit))
                if len(pmr_per_kc) > 0:
                    week_pmr[week].append(np.mean(pmr_per_kc))
                    
        results_pmr = {
            d: {"mean": np.mean(vals), "std": np.std(vals)}
            for d, vals in week_pmr.items() if vals
        }
        return results_pmr
    
    def compute_global_pmr(self, results_pmr, retention_pmr):
        global_pmr = {}
        for week, vals in results_pmr.items():
            
            global_pmr[week] = vals
        
        week_end_lr = max(results_pmr.keys()) +1
        for week, vals in retention_pmr.items():
            global_pmr[week_end_lr+week] = vals
        
        return global_pmr
        

    def simulate_choffin(self, params, r=3):
        curriculum = self.build_curriculum(self.kcs, self.weeks_to_simulate)
        queues_all = {}
        weekly_pmr = {}

        for student in self.students:
            queues = {}
            if hasattr(self.heuristic, 'reset'):
                self.heuristic.reset()
            kcs_introduced = []
            
            for week in range(self.weeks_to_simulate):
                new_kcs = curriculum.get(week, [])
                for kc in new_kcs:
                    if kc not in queues:
                        queues[kc] = {"wins": OurQueue(), "attempts": OurQueue()}
                    queues[kc]["attempts"].push(week * 7 * 24 * 3600)
                t_week = week * 7 * 24 * 3600
                kcs_introduced.extend(new_kcs)
                if week > 0:
                    
                    for rep in range(r):
                        item, kcs_item = self.heuristic.HeuristicTochooseItemfromQ(
                            week=week, kcs_introduced=kcs_introduced, q_mat_=self.qmat,
                            student=student, queues=queues, params=params,
                            t_current=t_week, items_per_kc=self.items_per_kc
                        )
                        if item is None:
                            continue

                        alpha_s = params["alpha_s"][student]
                        delta_j = params["delta_j"][item]
                        beta_sum = sum(params["beta_j"].get(k, 0) for k in kcs_item)
                        h_wins, h_attempts = 0.0, 0.0
                        for k in kcs_item:
                            if k not in queues:
                                queues[k] = {"wins": OurQueue(), "attempts": OurQueue()}
                            cw = queues[k]["wins"].get_counters(t_week)
                            ca = queues[k]["attempts"].get_counters(t_week)
                            for i in range(5):
                                h_wins += params["theta_wins"].get(k, [0]*5)[i] * np.log(1 + cw[i])
                                h_attempts += params["theta_attempts"].get(k, [0]*5)[i] * np.log(1 + ca[i])

                        logit = alpha_s - delta_j + beta_sum + h_wins + h_attempts
                        p_correct = self.sigmoid(logit)
                        correct = int(p_correct > 0.5)

                        for k in kcs_item:
                            queues[k]["attempts"].push(t_week)
                            if correct:
                                queues[k]["wins"].push(t_week)
                            if hasattr(self.heuristic, 'update'):
                                self.heuristic.update(k, week)
                pmr_all = self.compute_pmr_all_kcs(student, params, queues, t_week)
                
                w_key = week + 1
                if w_key not in weekly_pmr:
                    weekly_pmr[w_key] = []
                weekly_pmr[w_key].append(pmr_all)

            queues_all[student] = queues

        results_pmr = {
            w: {"mean": np.mean(vals), "std": np.std(vals)}
            for w, vals in weekly_pmr.items() if vals
        }
        retention_pmr = self.evaluate_retention(params, queues_all, curriculum)
        global_pmr = self.compute_global_pmr(results_pmr, retention_pmr)
        return results_pmr, None, retention_pmr, global_pmr

    def compute_pmr_all_kcs(self, student, params, queues, t_eval):
        
        pmr_list = []
        alpha_s = params["alpha_s"][student]
        for kc in self.kcs:
            beta = params["beta_j"].get(kc, 0)
            items = self.items_per_kc.get(kc, [])
            if len(items) == 0:
                delta_j=-1
            else:
                item=np.random.choice(items)
                delta_j = params["delta_j"][item]
            if kc in queues:
                cw = queues[kc]["wins"].get_counters(t_eval)
                ca = queues[kc]["attempts"].get_counters(t_eval)
            else:
                cw = [0] * 5
                ca = [0] * 5
            h = sum(
                params["theta_wins"].get(kc, [0]*5)[i] * np.log(1 + cw[i])
                + params["theta_attempts"].get(kc, [0]*5)[i] * np.log(1 + ca[i])
                for i in range(5)
            )
            # Choffin utilise delta=-1 fixe pour le PMR
            logit = alpha_s - delta_j + beta + h
            #print(f"Student {student}, KC {kc},cw: {cw}, ca: {ca}, logit: {logit:.2f}, PMR: {self.sigmoid(logit):.4f}")
            pmr_list.append(self.sigmoid(logit))
        #print(f"Student {student}, PMR all KCs: {np.mean(pmr_list):.4f}")
        return np.mean(pmr_list)