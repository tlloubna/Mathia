
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
class DifficultyANALYZER:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.item_difficulty = None
        self.scaler = MinMaxScaler()
    #Item
    def ScoreItemDiff(self, min_attempts: int = 30):
        # Agrégation par item
        item_stats = self.data.groupby('item_id').agg({
            'corrects': 'mean',                 
            'duration': 'mean',                 
            'hints': 'mean',                    
            'correct_first_attempt': 'mean',    
            'user_id': 'count'                  
        })

        item_stats.columns = [ 'success_rate','avg_time','avg_hints','avg_first_attempt','count']

        item_stats = item_stats[item_stats['count'] >= min_attempts] # nombre total minimale de tentatives pour cet item

        metrics = ['success_rate', 'avg_time', 'avg_hints', 'avg_first_attempt']
        item_stats_norm = item_stats.copy()
        item_stats_norm[metrics] = self.scaler.fit_transform(item_stats[metrics]) #normaliser

        item_stats['difficulty_score'] = ((1 - item_stats_norm['success_rate']) * 0.40 + item_stats_norm['avg_time'] * 0.20 
                                          +item_stats_norm['avg_hints'] * 0.20 + (1 - item_stats_norm['avg_first_attempt']) * 0.20 )
        item_stats['difficulty_level'] = pd.cut(item_stats['difficulty_score'], bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=['Easy', 'Medium', 'Difficult', 'Very Difficult']
        )

        self.item_difficulty = item_stats.sort_values('difficulty_score', ascending=False)
        return self.item_difficulty
    
    def ScoreKcDiff(self, min_attempts: int = 50):

            df = self.data.copy()
            df["KC"] = df["KC"].astype(str).str.split("~~")
            df = df.explode("KC")
            kc_stats = df.groupby("KC").agg({
                'correct_first_attempt': 'mean',   
                'duration': 'mean',               
                'hints': 'mean',                
                'corrects': 'mean',               
                'user_id': 'count'          #intercation between student and KC       
            })

            kc_stats.columns = ['success_rate', 'avg_time','avg_hints','avg_attempts','count']
            kc_stats = kc_stats[kc_stats['count'] >= min_attempts]
            metrics = ['success_rate', 'avg_time', 'avg_hints', 'avg_attempts']
            kc_stats_norm = kc_stats.copy()
            kc_stats_norm[metrics] = self.scaler.fit_transform(kc_stats[metrics])

            # Score de difficulté
            kc_stats['difficulty_score'] = (
                (1 - kc_stats_norm['success_rate']) * 0.40 +
                kc_stats_norm['avg_time'] * 0.20 +
                kc_stats_norm['avg_hints'] * 0.20 +
                kc_stats_norm['avg_attempts'] * 0.20
            )
            kc_stats['difficulty_level'] = pd.cut(
                kc_stats['difficulty_score'],
                bins=[0, 0.25, 0.50, 0.75, 1.0],
                labels=['Easy', 'Medium', 'Difficult', 'Very Difficult']
            )

            self.kc_difficulty = kc_stats.sort_values('difficulty_score', ascending=False)
            return self.kc_difficulty

