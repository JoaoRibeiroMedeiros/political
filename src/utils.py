
import math 
import pandas as pd
from datetime import timedelta, datetime
from src.crop import (crop_all_statements,crop_all_statements_per_politician, 
                      crop_statements_from_t0_to_t, crop_statements_until_t)

def binomial_coefficient(n, k):
    return math.comb(n, k)


def days_from_td(delta):
    total_seconds = delta.total_seconds()
    days = total_seconds / (24 * 3600)  
    return days

def lags_from_td(delta, lag):
    total_seconds = delta.total_seconds()
    days = total_seconds / (lag * 24 * 3600)  
    return days


class PoliticianStatements :

    def __init__(self, path, deputados_path, simulate = False, simulate_time = None):

        if not simulate:

            self.df = pd.read_csv(path)
            self.df = self.df.sort_values(by=['time'])
            self.df.time = pd.to_datetime(self. df.time)

            if simulate_time :
                self.df = self.df[self.df['time']< simulate_time]

            self.N = len(set(self.df.Id_politico))
            self.deputados = pd.read_csv(deputados_path)
            
    def head(self):

        return self.df.head()
    
    def get_politician(self, _id):
        return self.deputados[self.deputados['Id_politico'] == _id]
    
    def get_all_statements_per_politician(self, _id):
        return crop_all_statements_per_politician(self.df, _id)
    
    def get_politicians(self):

         # id_politicos = [id_politico for statements, id_politico in crop_statements_until_t(self.df, self.times.iloc[-1])]  ?
        ids = list(int(i) for i in self.df['Id_politico'].unique())

        return ids
    
    def get_politician_names(self):

        ids = self.get_politicians()
        names = [self.deputados[self.deputados['Id_politico']==_id]['NOME'].values[0] for _id in ids]

        return names
    
    def get_statements_num_and_sum(self, _id):
        print(_id, len(self.df[self.df['Id_politico']==_id]), sum(self.df[self.df['Id_politico']==_id]['isFavorable']))
    

    def get_rates(self, lag):

        ids = self.get_politicians()

        from_id_to_df = {id : self.df[self.df['Id_politico'] == id] for id in ids}
        from_id_to_rate = {id : 1 / (lags_from_td(np.mean(from_id_to_df[id].time.diff()), lag)) for id in ids}
        from_id_to_rate = {id : 0 if np.isnan(from_id_to_rate[id]) else from_id_to_rate[id] for id in ids}

        return from_id_to_rate
    
    def get_expected_number_of_posts_until_reckoning(self, days_to_reckoning):

        ids = self.get_politicians()
        from_id_to_rate = self.get_rates()
        from_id_to_expected_number_of_posts_until_reckoning =  {id: round(from_id_to_rate[id]*days_to_reckoning)  for id in ids}
    
        return from_id_to_expected_number_of_posts_until_reckoning

    def count_lags(start_datetime, end_datetime, lagsize):

        current_datetime = start_datetime
        count = 0

        while current_datetime < end_datetime:

            current_datetime += lagsize
            count += 1

        return count
    
    def get_timeframes(self, lag, day_of_reckoning):

        # current timeframe size
        total_distance =  (self.df.time.iloc[-1] - self.df.time.iloc[0]).total_seconds() 

        # timeframe size to reckoning
        total_distance_to_reckoning = (day_of_reckoning - self.df.time.iloc[0]).total_seconds() 

        # number of lag time intervals inside of current timeframe
        nlags =  round(total_distance/timedelta(days=lag).total_seconds())

        # number of lag time intervals inside of current timeframe
        lags_to_reckoning = round(total_distance_to_reckoning/timedelta(days=lag).total_seconds()) # unit is lags

        self.nlags = nlags

        self.lags_to_reckoning = lags_to_reckoning

        times = pd.Series([self.df.time.iloc[0] + timedelta(days=lag)*i for i in range(nlags)] )

        self.times = times

        return self
    