import numpy as np 
from src.Process.Simulation.utils import (compute_pmr_for_kc,classify_mastery)
from utils.this_queue import OurQueue
import pandas as pd 
import copy
import time 
class Evaluator:
    def __init__(self):
        pass


    def history_to_queues(self,history_df, student):
        df_std = history_df[history_df["user_id"] == student].sort_values("timestamp")
        queues = {}
        kcs_introduced = []
        for _, row in df_std.iterrows():
            kc = int(row["KC"])
            ts = float(row["timestamp"])
            correct = int(row["correct"])
            if kc not in queues:
                queues[kc] = {"wins": OurQueue(), "attempts": OurQueue()}
                kcs_introduced.append(kc)
            queues[kc]["attempts"].push(ts)
            if correct == 1:
                queues[kc]["wins"].push(ts)
        return queues, kcs_introduced
        
    def history_to_last_review(self,history_df, student, seconds_per_week=7*24*3600):
        df_std = history_df[history_df["user_id"] == student]
        last_review = {}
        for kc in df_std["KC"].unique():
            df_kc = df_std[df_std["KC"] == kc]
            last_ts = df_kc["timestamp"].max()
            last_review[int(kc)] = int(last_ts // seconds_per_week)
        return last_review

    def run_review_session(self,heuristic, queues, kcs_introduced, params, qmat,
                        student, items_per_kc, all_kcs,
                        t_start, t_max_minutes, current_week, rng):
        t_current = t_start
        t_end = t_start + t_max_minutes * 60
        kcs_chosen_this_session = []

        alpha_s = params["alpha_s"][student]
        decision_time = 0.0     
        n_decisions = 0 
        while t_current < t_end:
            pmr_per_kc = {
                kc: compute_pmr_for_kc(kc, params, queues, t_current,
                                        alpha_s, items_per_kc)
                for kc in all_kcs
            }

            try:
                t0 = time.perf_counter() 
                item, kcs_picked = heuristic.HeuristicTochooseItemfromQ(
                    week=current_week,
                    kcs_introduced=kcs_introduced,
                    q_mat_=qmat,
                    student=student,
                    queues=queues,
                    params=params,
                    t_current=t_current,
                    items_per_kc=items_per_kc,
                    dictPkcs=pmr_per_kc,
                    kc_idx_to_name=None,
                    kc_name_to_idx=None,
                )
                decision_time += time.perf_counter() - t0   
                n_decisions += 1
            except Exception as e:
                print(f"[!] Heuristique a échoué: {e}")
                break

            if item is None or not kcs_picked:
                break

            delta_j = params["delta_j"][int(item)]
            beta_sum = sum(params["beta_j"].get(kc, 0) for kc in kcs_picked)
            h_wins, h_attempts = 0.0, 0.0
            for kc in kcs_picked:
                if kc not in queues:
                    queues[kc] = {"wins": OurQueue(), "attempts": OurQueue()}
                cw = queues[kc]["wins"].get_counters(t_current)
                ca = queues[kc]["attempts"].get_counters(t_current)
                for i in range(5):
                    h_wins += params["theta_wins"][kc][i] * np.log(1 + cw[i])
                    h_attempts += params["theta_attempts"][kc][i] * np.log(1 + ca[i])

            logit = alpha_s - delta_j + beta_sum + h_wins + h_attempts
            p_correct = 1 / (1 + np.exp(-logit))
            correct = int(rng.random() < p_correct)
            for kc in kcs_picked:
                queues[kc]["attempts"].push(t_current)
                if correct:
                    queues[kc]["wins"].push(t_current)
                if hasattr(heuristic, "update"):
                    heuristic.update(kc, current_week)

            kcs_chosen_this_session.extend(int(kc) for kc in kcs_picked)

            exo_duration = rng.integers(2, 8) * 60
            t_current += exo_duration

        return kcs_chosen_this_session,decision_time,n_decisions


    def build_comparison_table_multiweek(self,df_hist, qmat, params, heuristics, students,
                                        items_per_kc, all_kcs,
                                        n_review_weeks=4, t_max_minutes=15,
                                        hist_n_days=10, seed=42,
                                        kcs_mode="all"):
        rows = []
        pmr_evolution_all = {}

        t_review_start = hist_n_days * 24 * 3600
        seconds_per_week = 7 * 24 * 3600
        week_offset = hist_n_days // 7

        for std in students:
            queues_init, kcs_init = self.history_to_queues(df_hist, student=std)
            last_review_init = self.history_to_last_review(df_hist, student=std)
            alpha_s = params["alpha_s"][std]

            if kcs_mode == "all":
                kcs_introduced_h_base = list(all_kcs)
            elif kcs_mode == "introduced":
                kcs_introduced_h_base = list(kcs_init) if kcs_init else list(all_kcs)
            else:
                raise ValueError(f"kcs_mode inconnu : {kcs_mode}")

            pmr_initial = {
                kc: compute_pmr_for_kc(kc, params, queues_init,
                                        t_review_start, alpha_s, items_per_kc)
                for kc in all_kcs
            }

            choices_per_heuristic = {name: {} for name in heuristics}
            pmr_history_per_heuristic = {}
            timing_rows = []
            for name, heuristic in heuristics.items():
                queues_h = copy.deepcopy(queues_init)
                kcs_introduced_h = list(kcs_introduced_h_base)

                if hasattr(heuristic, "reset"):
                    heuristic.reset()
                if hasattr(heuristic, "last_review"):
                    heuristic.last_review = dict(last_review_init)

                local_rng = np.random.default_rng(
                    seed + abs(hash((name, std))) % 100000
                )

                pmr_history_per_heuristic[name] = {}
                pmr_history_per_heuristic[name][-1] = {
                    kc: compute_pmr_for_kc(kc, params, queues_h, t_review_start,
                                            alpha_s, items_per_kc)
                    for kc in all_kcs
                }
                total_decision_time = 0.0
                total_decisions = 0

                for w in range(n_review_weeks):
                    current_week = week_offset + w
                    t_start = t_review_start + w * seconds_per_week

                    chosen,dt,nd = self.run_review_session(
                        heuristic=heuristic, queues=queues_h,
                        kcs_introduced=kcs_introduced_h,
                        params=params, qmat=qmat,
                        student=std, items_per_kc=items_per_kc,
                        all_kcs=all_kcs,
                        t_start=t_start, t_max_minutes=t_max_minutes,
                        current_week=current_week, rng=local_rng,
                    )
                    choices_per_heuristic[name][w] = chosen

                    t_end_week = t_start + seconds_per_week
                    pmr_history_per_heuristic[name][w] = {
                        kc: compute_pmr_for_kc(kc, params, queues_h, t_end_week,
                                                alpha_s, items_per_kc)
                        for kc in all_kcs
                    }
                    choices_per_heuristic[name][w] = chosen
                    total_decision_time += dt
                    total_decisions += nd
                timing_rows.append({
                            "student": std,
                            "heuristique": name,
                            "temps_total_s": total_decision_time,
                            "n_decisions": total_decisions,
                            "temps_moyen_ms": (total_decision_time / total_decisions * 1000)
                                            if total_decisions else 0.0,
                        })
            pmr_evolution_all[std] = pmr_history_per_heuristic

            for kc in all_kcs:
                row = {
                    "student": std,
                    "KC": kc,
                    "niveau_initial": classify_mastery(pmr_initial[kc]),
                    "pmr_initial": round(pmr_initial[kc], 3),
                    "last_review_initial": last_review_init.get(kc, None),
                }
                for name in heuristics.keys():
                    total = 0
                    for w in range(n_review_weeks):
                        n_choices = choices_per_heuristic[name][w].count(kc)
                        row[f"{name}_w{w}"] = n_choices
                        total += n_choices
                    row[f"{name}_total"] = total
                    pmr_final_kc = pmr_history_per_heuristic[name][n_review_weeks - 1][kc]
                    row[f"{name}_pmr_final"] = round(pmr_final_kc, 3)

                rows.append(row)

        return pd.DataFrame(rows), pmr_evolution_all,pd.DataFrame(timing_rows)
