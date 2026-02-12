import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.graphics.PlotOutills as PO
import src.Analysis.DifficultyAnalyzer as DA
#data Algebre05
pathAlgebre="/home/loubna/Code Projet Mathia/Mathia/data/algebra05/data.txt"
#data Bridge06
pathbridge = "/home/loubna/Code Projet Mathia/Mathia/data/bridge_algebra06/data.txt"
choice =2
if choice ==1: 
    stdmodel=SD.StudentDATA(file=pathAlgebre)
else :
    stdmodel=SD.StudentDATA(file=pathbridge)
stdmodel.loadData(Display=True)
PlotO=PO.PlotOUTILS()
DifScore=DA.DifficultyANALYZER(stdmodel.data)
Item_diff=DifScore.ScoreItemDiff(min_attempts=20)
Kc_diff=DifScore.ScoreKcDiff(min_attempts=60)

PlotO.PlotScoreDifficulty(score_diff=Item_diff)
PlotO.PlotScoreDifficulty(score_diff=Kc_diff)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
