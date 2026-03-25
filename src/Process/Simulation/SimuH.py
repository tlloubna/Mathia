import src.Process.DAS3H as das3H
from utils.this_queue import OurQueue
import src.datamodel.Studentdata as STD
from collections import defaultdict
import src.Process.Simulation.utils as simu_utils
import random
import pandas as pd
import numpy as np
class SimulationH():
    def __init__(self,data:pd.DataFrame=None,model:das3H.DAS3HModel=None,qmat=None,heuristic=None,kc_list=None):

        self.data=data
        self.model=model
        self.qmat=qmat
        self.heuristic=heuristic
        self.StudentModel=STD.StudentDATA()
        self.seed=42
        self.kc_list=kc_list 
    def SelectRandomstudent(self,nb_stds=500):
        #fixer la graine pour la reproductibilité
        random.seed(self.seed)
        students = random.sample(list(self.data["user_id"].unique()), nb_stds)
        students = [s for s in students if len(self.data[self.data["user_id"]==s])>30]
        print("Number of students selected:", len(students))
        return students
    def getKClist(self):
        kc_list= []
        for kc in self.data["KC"].unique():
            for elt in kc.split("~~"):
                kc_list.append(elt)
        return kc_list
    
    def buildQfromData(self,student_id=None,empty=True):
        if empty:
            return defaultdict(lambda:OurQueue())
        q=defaultdict(lambda:OurQueue())
        data_std=self.data[self.data["user_id"]==student_id] if student_id is not None else self.data
        sorted_data=data_std.sort_values(by="timestamp")
        kc_list=self.getKClist()
        for _, row in sorted_data.iterrows():
            t=row["timestamp"]
            kcs=str(row["KC"]).split("~~")
            for kc in kcs:
                kc=kc.strip()
                if kc not in kc_list:
                    continue
                q[(kc,"attempts")].push(t)
                if row["correct"]==1:
                    q[(kc,"wins")].push(t)
        return q
                
    """def Simulation(self, seuil=0.5, nb_students=500, reviews_per_step=3, review=True):
        students = self.SelectRandomstudent(nb_students)
        PMR_ap = np.zeros(10)
        PMR_ret = np.zeros(6)
        count = 0

        for std in students:
            count += 1
            print("Step :", count, "/", len(students))
            
            q_student = self.buildQfromData(std, empty=True)
            t0 = 0
            all_kcs_student = self.StudentModel.get_student_kcs(self.data, std)
            practiced_kcs = set()

            # Introduction de tous les KCs à t=0 avant la simulation
            kcs_to_introduce = all_kcs_student[:10]
            for kc in kcs_to_introduce:
                practiced_kcs.add(kc)
                q_student[(kc, "attempts")].push(0)
                q_student[(kc, "wins")].push(0)

            for week in range(10):
                t_courant = t0 + week * 7 * 24 * 3600

                if review:
                    self.heuristic.t_courant = t_courant
                    self.heuristic.student = std
                    self.heuristic.q = q_student
                    for _ in range(reviews_per_step):
                        result = self.heuristic.HeuristicTochooseItemfromQ()
                        if not result or len(result) == 0:
                            continue
                        item, kcs = result
                        if not kcs:
                            continue
                        p = np.mean([simu_utils.ComputePMR(self.model, std, kc,
                                                            t_courant, q_student, item)
                                    for kc in kcs])
                        outcome = 1 if p > 0.5 else 0
                        for kc in kcs:
                            q_student[(kc, "attempts")].push(t_courant)
                            if outcome == 1:
                                q_student[(kc, "wins")].push(t_courant)

                nb_mastered = sum([1 for kc in practiced_kcs
                                if simu_utils.ComputePMR(self.model, std, kc,
                                                            t_courant, q_student, None) >= seuil])
                PMR_ap[week] += nb_mastered / len(practiced_kcs)
                print(f"Student {std} - Week {week}: PMR_ap = {PMR_ap[week]:.4f}")

            for week in range(10, 16):
                t_courant = t0 + week * 7 * 24 * 3600
                nb_mastered = sum([1 for kc in practiced_kcs
                                if simu_utils.ComputePMR(self.model, std, kc,
                                                            t_courant, q_student, None) >= seuil])
                PMR_ret[week - 10] += nb_mastered / len(practiced_kcs)
                print(f"Student {std} - Week {week}: PMR_ret = {PMR_ret[week - 10]:.4f}")
        PMR_ap /= len(students)
        PMR_ret /= len(students)
        return list(PMR_ap), list(PMR_ret)"""
    def Simulation(self, nb_students=500, reviews_per_step=3, review=True):
        students = self.SelectRandomstudent(nb_students)
        PMR_ap = np.zeros(10)
        PMR_ret = np.zeros(6)

        # Initialisation : construire les queues et introduire les KCs pour chaque étudiant
        kc_to_idx = {kc: i for i, kc in enumerate(self.kc_list)}
        
        students_data = {}
        for std in students:
            # Queues vides
            q_student = self.buildQfromData(std, empty=True)
            
            # KCs dédoublonnés et avec items
            all_kcs = list(dict.fromkeys(self.StudentModel.get_student_kcs(self.data, std)))
            kcs_to_introduce = []
            for kc in all_kcs:
                if kc in kc_to_idx and np.sum(self.qmat[:, kc_to_idx[kc]]) > 0:
                    kcs_to_introduce.append(kc)
                if len(kcs_to_introduce) == 10:
                    break
            
            students_data[std] = {
                "q": q_student,
                "kcs": kcs_to_introduce,
                "introduced": []  # KCs introduits progressivement
            }

        # Phase apprentissage
        for week in range(10):
            t_courant = week * 7 * 24 * 3600
            PMR_semaine = []

            for std in students:
                q_student = students_data[std]["q"]
                kcs_to_introduce = students_data[std]["kcs"]
                introduced = students_data[std]["introduced"]

                # Introduction du KC de la semaine
                if week < len(kcs_to_introduce):
                    kc_new = kcs_to_introduce[week]
                    introduced.append(kc_new)
                    q_student[(kc_new, "attempts")].push(t_courant)
                    q_student[(kc_new, "wins")].push(t_courant)

                # Révision (seulement à partir de la semaine 1)
                if review and week > 0:
                    self.heuristic.t_courant = t_courant
                    self.heuristic.student = std
                    self.heuristic.q = q_student
                    for _ in range(reviews_per_step):
                        result = self.heuristic.HeuristicTochooseItemfromQ(week=week, kcs_introduced=introduced)
                        if not result or len(result) == 0:
                            continue
                        item, kcs = result
                        if not kcs:
                            continue
                        p = np.mean([simu_utils.ComputePMR(self.model, std, kc,
                                                            t_courant, q_student, item)
                                    for kc in kcs])
                        outcome = 1 if p > 0.5 else 0
                        for kc in kcs:
                            q_student[(kc, "attempts")].push(t_courant)
                            if outcome == 1:
                                q_student[(kc, "wins")].push(t_courant)

                # PMR de cet étudiant = moyenne des P sur les KCs introduits
                if len(introduced) > 0:
                    pmr_std = np.mean([simu_utils.ComputePMR(self.model, std, kc,
                                                            t_courant, q_student, None)
                                    for kc in introduced])
                    PMR_semaine.append(pmr_std)

            # PMR globale semaine = moyenne sur les étudiants
            if len(PMR_semaine) > 0:
                PMR_ap[week] = np.mean(PMR_semaine)
            print(f"Semaine {week} apprentissage : PMR = {PMR_ap[week]:.4f}")

        # Phase rétention
        for week in range(10, 16):
            t_courant = week * 7 * 24 * 3600
            PMR_semaine = []

            for std in students:
                q_student = students_data[std]["q"]
                introduced = students_data[std]["introduced"]

                # Pas de révision — queues figées
                if len(introduced) > 0:
                    pmr_std = np.mean([simu_utils.ComputePMR(self.model, std, kc,
                                                            t_courant, q_student, None)
                                    for kc in introduced])
                    PMR_semaine.append(pmr_std)

            if len(PMR_semaine) > 0:
                PMR_ret[week - 10] = np.mean(PMR_semaine)
            print(f"Semaine {week} rétention : PMR = {PMR_ret[week-10]:.4f}")

        return list(PMR_ap), list(PMR_ret)