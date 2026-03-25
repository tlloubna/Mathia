import numpy as np
import os
from sklearn.model_selection import KFold

class CrossValid:
    def __init__(self,data,name_dataset,nb_folds=5,perc_init=0.2,data_Folder="data/",random_state=42):
        self.data=data
        self.name_dataset=name_dataset
        self.nb_folds=nb_folds
        self.perc_init=perc_init
        self.data_Folder=data_Folder
        self.random_state=random_state

    def prepare_folds(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def saveStrongestFolds(self):
        all_users=self.data['user_id'].unique()
        kf = KFold(n_splits=self.nb_folds, shuffle=True, random_state=self.random_state)
        path=os.path.join(self.data_Folder,self.name_dataset,"strongest","folds")
        self.prepare_folds(path)
        for i , (train,test) in enumerate(kf.split(all_users)):
            list_of_test_ids=[]
            
            for user_id in all_users[test]:
                print("Process strong",i,"Step",user_id,"/",len(all_users))
                list_of_test_ids+=list(self.data[self.data['user_id']==user_id].index)


            np.save(os.path.join(path,f"test_fold_{i}.npy"),np.array(list_of_test_ids))
            print(f"Fold {i} saved with {len(list_of_test_ids)} test samples." )
        print("All strongest folds saved successfully.")


    def savePseudoStrongFolds(self):
        all_users=self.data['user_id'].unique()
        kf = KFold(n_splits=self.nb_folds, shuffle=True, random_state=self.random_state)
        path=os.path.join(self.data_Folder,self.name_dataset,"pseudo_strong","folds")
        self.prepare_folds(path)
        for i , (train,test) in enumerate(kf.split(all_users)):
            print(f"Processing fold {i}","//",self.nb_folds)
            list_of_test_ids=[]
            for user_id in all_users[test]:
                print("Process pseudo",i,"Step",user_id,"/",len(all_users))
                fold=self.data[self.data['user_id']==user_id].sort_values(by='timestamp').index
                list_of_test_ids+=list(fold[round(self.perc_init*len(fold)):])
            np.save(os.path.join(path,f"test_fold_{i}.npy"),np.array(list_of_test_ids))
            print(f"Fold {i} saved with {len(list_of_test_ids)} test samples." )
        print("All pseudo-strong folds saved successfully.")
        
    def getfold(self, fold_id, split_type="strongest"):
        path = os.path.join(self.data_Folder, self.name_dataset, 
                            split_type, "folds", f"test_fold_{fold_id}.npy")
        test_ids = np.load(path)
        # Utilise les index du df directement avec set difference — beaucoup plus rapide
        test_ids_set = set(test_ids)
        train_ids = np.array(list(set(self.data.index) - test_ids_set))
        return train_ids, test_ids

