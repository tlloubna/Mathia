#this paragraph contains instructions to add the parent directory in the python path.
import os,sys
extra_path=os.path.join(os.path.dirname(__file__), "..")
try:
    sys.path.index(extra_path)
except:
    sys.path.append(extra_path)

import src.datamodel.Studentdata as SD
import src.Process.DAS3H as DAS3H
import src.datamodel.Historydata as HIS
import src.graphics.PlotOutills as Plot
from scipy import sparse
import pandas as pd 
#data Algebre05
pathAlgebre="/home/loubna/Code Projet Mathia/Mathia/data/algebra05/data.txt"
#data Bridge06
pathbridge = "/home/loubna/Code Projet Mathia/Mathia/data/bridge_algebra06/data.txt"

plot=Plot.PlotOUTILS()
choice =2
if choice ==1: 
    stdmodel=SD.StudentDATA(file=pathAlgebre)
else :
    stdmodel=SD.StudentDATA(file=pathbridge)

df,Q=stdmodel.loadData(Display=True,min_intercation=30,n_students=50)
#stdmodel.CleanData(Nbofseenitem=100,NbofseenKc=500)

#saveQ 
folder_path = os.path.join("data", "bridge_algebra06")
#sparse.save_npz(folder_path + "/q_mat_50std.npz", sparse.csr_matrix(Q))
#df.to_csv(folder_path + "/preprocessed_data_50std.csv", index=False)

df = pd.read_csv(folder_path + "/preprocessed_data_50std.csv", sep=",")
qmat = sparse.load_npz(folder_path + "/q_mat_50std.npz").toarray()
his=HIS.HistoryDATA(stdmodel=stdmodel)
X=his.ComputeHistoryFeaturesTWKC(Q_mat=qmat,df=df)
sparse.save_npz(folder_path+"X-student50Bridge.npz", X)
##DAS3H
#plot.plot_Q_matrix(Q=qmat,max_items=4000)
#X=sparse.load_npz(folder_path+"X-student50Bridge.npz")

user_ids = his.user_ids      # liste ordonnée des user_id  (depuis enc_users.categories_[0])
item_ids = his.item_ids      # liste ordonnée des item_id  (depuis enc_items.categories_[0])
kc_list  = stdmodel.KComp
model = DAS3H.DAS3HModel(C=1.0) 
results = model.fit(
    X,
    user_ids = user_ids,
    item_ids = item_ids,
    kc_list  = kc_list,
    n_tw     = 5           # doit correspondre à NB_OF_TIME_WINDOWS dans HistoryDATA
)
#plot.PlotROC(TPR=results.get("TPR"),FPR=results.get("FPR"),AUC=results.get("AUC"))


params = model.get_params()


plot.PlotStudentTrajectory(params, df, user_id=3, top_n_kc=5)

# Pour tester plusieurs élèves rapidement
for uid in [0, 1, 5, 12]:
    plot.PlotStudentTrajectory(params, df, user_id=uid, top_n_kc=3)

    
# 1. P(correct) décroît-elle bien avec la difficulté ?
plot.PlotProbVsDifficulty(params)

# 2. Comment sont distribués les élèves et les items ?
plot.PlotDistribParams(params)

# 3. Courbes d'oubli theta_wins et theta_attempts par KC × fenêtre temporelle
plot.PlotForgettingCurves(params, top_n=8)

# 4. Contribution mémoire log(1+n)×theta en fonction du nb de succès
plot.PlotMemoryEffect(params, top_n=6, tw_idx=1)  # tw_idx=1 → fenêtre 1 jour

# 5. Simulation d'un élève spécifique sur un item
kc_de_item = df[df["item_id"] == 5]["KC"].iloc[0].split("~~")
plot.PlotStudentLearning(params, user_id=0, item_id=5, kc_names=kc_de_item)

# 6. Heatmap KC × fenêtre temporelle
plot.PlotThetaHeatmap(params, top_n=20)

# 7. Dashboard complet en un seul appel ← le plus utile
plot.PlotDAS3HDashboard(params, top_n_kc=6)

print("done")


