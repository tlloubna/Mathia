import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.graphics.PlotOutills as PO
import src.Process.IRTModel as IRT


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


stdmodel.loadData(Display=False)
IRTmdl=IRT.IRTMODEL(DataStudent=stdmodel)
lr=0.001
epoches=1000
IRTmdl.fit(lr=lr,epoches=epoches)

stdalpha=IRTmdl.getStdability()
stddelta=IRTmdl.getKcDiff()


