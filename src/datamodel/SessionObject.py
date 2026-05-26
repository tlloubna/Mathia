import src.datamodel.Historydata as hist
import src.datamodel.Studentdata as std
import src.Process.DAS3H as das3H
import src.Process.Estimator.AlphaEstimator as Ae
import src.Process.IRTModel as irt 
import src.Process.Simulation.SimuH as simh 

class Session:
    def __init__(self):
        self.ListStdmodel:list[std.StudentDATA]=[]
        self.simulations:list[simh.SimulationH]=[]
        self.Modelpmr:list=[]

    def fillSession(self,ListStdmodel=[],simulation=[],modelpmr=[]):
        self.ListStdmodel=ListStdmodel
        self.simulations=simulation
        self.Modelpmr=modelpmr
    def loadSessionFromdisk(self,KT_filepath=""):
        pass

    def saveSessionTodisk(self,Kt_filepath=""):
        pass

    def describeSessionParameters(self):
        txt=""
        txt+="Session description :\n"
        txt+="--List of dataSet: \n"

        for data in self.ListStdmodel:
            txt+="\t-"+data.name +"\n"
        txt+="-- List of simulation: \n"
        for sim in self.simulations:
            txt+="\t-"+sim.name +"\n"
        for model in self.Modelpmr: 
            txt+="\t-"+model.name +"\n"
        return txt 
    
    def CleanSession(self):
        self.ListStdmodel=[]
        self.simulations=[]
        self.Modelpmr=[]
    