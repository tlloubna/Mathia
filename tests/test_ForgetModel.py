import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.graphics.PlotOutills as PO
import src.Analysis.DifficultyAnalyzer as DA
import src.Process.ForgettingModel as FGM
import random 


#data Algebre05
pathAlgebre="/home/loubna/Code Projet Mathia/Mathia/data/algebra05/data.txt"
#data Bridge06
pathbridge = "/home/loubna/Code Projet Mathia/Mathia/data/bridge_algebra06/data.txt"
plotO=PO.PlotOUTILS()
choice =2
if choice ==1: 
    stdmodel=SD.StudentDATA(file=pathAlgebre)
else :
    stdmodel=SD.StudentDATA(file=pathbridge)
stdmodel.loadData(Display=True)

Fmodel=FGM.ForgettingMODEL(DataStudent=stdmodel)
Fg=Fmodel.EstimateFG()

plotO.plot_all_forgetting_curves(Fg)


print("done !!!")