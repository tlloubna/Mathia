import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.graphics.PlotOutills as PO
import src.Process.ForgettingModel as FGM

#data Algebre05
pathAlgebre="/home/loubna/Code Projet Mathia/Mathia/data/algebra05/data.txt"
#data Bridge06
pathbridge = "/home/loubna/Code Projet Mathia/Mathia/data/bridge_algebra06/data.txt"

choice =2
if choice ==1: 
    stdmodel=SD.StudentDATA(file=pathAlgebre)
else :
    stdmodel=SD.StudentDATA(file=pathbridge)
PlotO=PO.PlotOUTILS()

stdmodel.loadData(Display=True,min_intercation=30)

stdmodel.CleanData(Nbofseenitem=30,NbofseenKc=100)

dicSI, dicKI = stdmodel.dicSIandIK()

Fmodel=FGM.ForgettingMODEL(DataStudent=stdmodel)
Fg=Fmodel.EstimateFG()

PlotO.plot_all_forgetting_curves(Fg)
#Plot
kc_presence = {}
for item, kcs in dicKI.items():
    for kc in kcs:
        kc_presence[kc] = kc_presence.get(kc, 0) + 1


item_presence = {}
for std, items in dicSI.items():
    for item in items:
        item_presence[item] = item_presence.get(item, 0) + 1 


PlotO.plotpresence(kc_presence,vars=["kc","nb of item"])
PlotO.plotpresence(item_presence,vars=["item","nb of std"])

print("Done!!!!")
