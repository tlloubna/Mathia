#this paragraph contains instructions to add the parent directory in the python path.
import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)



import src.graphics.Exploredata as EXD


#data Algebre05
pathAlgebre="/home/loubna/Code Projet Mathia/Mathia/data/algebra05/data.txt"
#data Bridge06
pathbridge = "/home/loubna/Code Projet Mathia/Mathia/data/bridge_algebra06/data.txt"

expd1=EXD.ExploreDATA(file=pathAlgebre)
expd2=EXD.ExploreDATA(file=pathbridge)

data1=expd1.loadData(Display=True)
data2=expd2.loadData(Display=True)

"""expd1.MatrixCorr()
expd2.MatrixCorr()"""