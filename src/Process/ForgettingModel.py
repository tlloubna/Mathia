import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class ForgettingMODEL:
    def __init__(self,data:pd.DataFrame=None):
        self.df=data
    


    def StrongMemory(self):
        self.df = self.data.copy()
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        success_rate = self.df.groupby(['user_id', 'item_id'])['correct_first_attempt'].mean()
        num_practices = self.df.groupby(['user_id', 'item_id']).size()
        last_review = self.df.groupby(['user_id', 'item_id'])['start_time'].max()
        now = pd.Timestamp.now()
        time_since_last = (now - last_review).dt.total_seconds() / 3600  # heures
        S0 = 24
        w1 = 48
        w2 = 4
        w3 = 0.2
        memory_strength = (
            S0
            + w1 * success_rate
            + w2 * num_practices
            - w3 * time_since_last
        )
        memory_strength = memory_strength.clip(lower=1, upper=240)

        return memory_strength

        

    def ForgetModel1(self,t:float=4,S:float=52):
        return np.exp(-t/S)
    def ForgetModel2(self,t:float=4,k:float=10):
        return 100/(1+k*t)
    



        