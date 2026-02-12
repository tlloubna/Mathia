import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.graphics.PlotOutills as PO


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

stdmodel.loadData(Display=True)
stdmodel.build_Q_matrix()
PlotO.plot_Q_matrix(Q=stdmodel.Q,KComp=stdmodel.KComp,max_items=5000)


print("done!!!! ")

