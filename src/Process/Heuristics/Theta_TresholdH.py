import src.Process.DAS3H as das3H
import src.datamodel.Studentdata as STD
import src.Process.Simulation.utils as simu_utils
import random 
class ThetaTresholdH():
    def __init__(self,model:das3H.DAS3HModel,StudentModel:STD.StudentDATA=None,student=None,listOfTheta:list[int]=[0.1,0.2,0.3,0.4,0.5,0.6]
                 ,data=None,qmat=None,kc_list=None,effort=1):
        self.model=model
        self.StudentModel=StudentModel
        self.student=student
        self.listOfTheta=listOfTheta
        self.data=data
        self.t_courant=data[data["user_id"]==student]["timestamp"].max() if student is not None else None
        self.qmat=qmat
        self.effort=effort 
        self.kc_list=kc_list
        self.q=None#Now we encode the effort : 1=low effort, 2=medium effort, 3=high effort
        #The effort signify the ressource allowed to the system to resolove the problem 
    def ComputePforallKC(self, q=None):
        listOfKC = self.StudentModel.get_student_kcs(self.data, self.student)
        PforallKC = {}
        for kc in listOfKC:
            PforallKC[kc] = simu_utils.ComputePMR(
                self.model, self.student, kc, 
                self.t_courant, q, item=None
            )
        return PforallKC
        
    def ChooseThetaFixed(self, theta=0.4):
        return theta
    def ChooseThetaWhoRespectEffort(self):
        #if the effort is low, choose a theta that requires less effort that mean
        #we will choose a kc that necessite less effort to be mastered==> choose max theta 
        if self.effort==1:
            return min(self.listOfTheta)
        elif self.effort==2:
            return self.listOfTheta[len(self.listOfTheta)//2]
        else:
            return max(self.listOfTheta)
    def ComputePMR(self,PforallKC,theta):
        #compute PMR for each kc 
        PMR_theta=0
        for kc, P in PforallKC.items():
            if P>=theta:
                PMR_theta+=1
        return PMR_theta/len(PforallKC)
        
    def DistanceFromTheta(self,PforallKC,theta):
        #compute the distance from theta for each kc 
        distance={}
        for kc, P in PforallKC.items():
            distance[kc]=abs(P-theta)
        return distance
    
    def listOfItemforKc(self, kc_list: list, q_mat):
        
        items = []
        for item_id in range(q_mat.shape[0]):
            if any(q_mat[item_id, kc] == 1 for kc in kc_list):
                items.append(item_id)
        return items
    
    def HeuristicTochooseItemfromQ(self, week=None, kcs_introduced=None):
    
        theta=self.ChooseThetaFixed() 
        
        PforallKC = self.ComputePforallKC(self.q)
        distances  = self.DistanceFromTheta(PforallKC, theta)
        sorted_KC  = sorted(distances.items(), key=lambda x: x[1])  

        kc_to_idx = {kc: i for i, kc in enumerate(self.kc_list)}
        anchor_kc = sorted_KC[0][0]
        acceptable_items = set(
                item_id for item_id in range(self.qmat.shape[0])
                if self.qmat[item_id, kc_to_idx[anchor_kc]] == 1
            )
        
        if len(acceptable_items) == 0:
            return []  
        selected_kc = [anchor_kc]
        for kc, distance in sorted_KC[1:]:
            if kc not in kc_to_idx:
                continue  
            kc_idx   = kc_to_idx[kc]
            items_kc = set(
                item_id for item_id in range(self.qmat.shape[0])
                if self.qmat[item_id, kc_idx] == 1
            )
            intersection = acceptable_items & items_kc
            if len(intersection) > 0:
                acceptable_items = intersection
                selected_kc.append(kc)

        if len(acceptable_items) > 1:
            return random.choice(list(acceptable_items)),selected_kc
        else:
            return list(acceptable_items)[0],selected_kc
                

