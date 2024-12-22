

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from src.crop import   (crop_statements_until_t,
                   crop_statements_from_t0_to_t, 
                   crop_statements_until_t_by_politician, 
                   crop_all_statements,crop_all_statements_per_politician)

from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
from itertools import product
from src.models import SimulateStatement, Model, PoliticianOpinion, PoliticiansOpinionInTime


class ProbabilityAnalysis: 

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
    
    def get_high_statement_volatility_politicians(self):
        high_volatility_ids = [int(i ) for i, v in self.from_politician_to_volatility.items() if 0.5 < v < 1]
        return high_volatility_ids

    def get_high_statement_volatility_info(self):

        high_volatility_ids = self.get_high_statement_volatility_politicians()

        high_volatility_politican_info = [self.get_politician(i) for i in high_volatility_ids]

        from_name_to_id_high_vol = {info['NOME'].values[0]: int(info['Id_politico'].values[0]) for info in high_volatility_politican_info}
        from_id_to_name_high_vol = {v: k for k, v in from_name_to_id_high_vol.items()}

        return  from_id_to_name_high_vol, from_name_to_id_high_vol, high_volatility_politican_info


    def get_statement_volatility(self):

        ids =  self.get_politicians()

        from_politician_to_volatility = {}

        for _id in ids:
            statements = self.get_all_statements_per_politician(_id)
            volatility = np.std(statements)
            from_politician_to_volatility[_id] = volatility

        self.from_politician_to_volatility = from_politician_to_volatility

        volatilities = list(from_politician_to_volatility.values())

        max_volatility = max(volatilities)
        max_volatility_index = volatilities.index(max_volatility)

        id_politician_with_max_volatility = ids[max_volatility_index]
        max_volatility_politician = self.get_politician(id_politician_with_max_volatility)

        self.statement_volatilities = volatilities
        self.max_volatility = max_volatility
        self.max_volatility_politician = max_volatility_politician

        return self

    def get_all_t0_t_whose_difference_is_lagsize(self, d , slide_lag = 24*3600 ):
        """
        Where m is posts per d based on politician rate
        """
    
        t0_t_pairs = []

        i = 0

        while self.times.iloc[0] + timedelta(seconds = i) < self.times.iloc[-1]:

            t0_t_pairs.append((self.times[0] + timedelta(seconds = i), self.times[0] + timedelta(seconds = i) + timedelta(seconds = 24*3600*d)))
            i += slide_lag

        return t0_t_pairs


    def get_post_trajectories_size_d_lags(self, d):
        """
        Get all different trajectories of opinions for a single politician.

        Parameters:
        - opinions_in_time: List of PoliticiansOpinionInTime instances.
        - politician_id: integer associated with politician.

        Returns:
        - A list of trajectories for the specified politician.
        """

        ids = self.get_politicians()

        from_politician_to_d_chopped_series = {i:[] for i in ids }

        t0_t_pairs = self.get_all_t0_t_whose_difference_is_lagsize(d)

        print('running sliding window')
        
        for (t0, t) in tqdm(t0_t_pairs):
            for elem in crop_statements_from_t0_to_t(self.df,t0,t):

                statements, id_politico = elem
                from_politician_to_d_chopped_series[int(id_politico)].append(statements)

        self.from_politician_to_d_chopped_series = from_politician_to_d_chopped_series

        return self

    def dynamic_programming(p_list, n):
        N = len(p_list)
        dp = [0] * (N + 1)
        dp[0] = 1
        
        for p in p_list:
            for i in range(N, 0, -1):
                dp[i] += p * (dp[i-1] - dp[i])
        
        return sum(dp[n:])
    
    def calculate_approval_probability(self,  l, delta, delta_method =  'dynamic'):

        ids = self.get_politicians()
        self = self.get_post_trajectories_size_d_lags(self.lags_to_reckoning)
        all_trajectories = 0
        approval_trajectories = 0
        list_probable_statements_after_t = list(self.from_politician_to_d_chopped_series.values())
        list_probable_statements_after_t = list(product(*list_probable_statements_after_t))      
        all_politician_statements = crop_all_statements(self.df)

        for statements_in_d in  tqdm(list_probable_statements_after_t):

            total_statements = all_politician_statements + statements_in_d
            politician_opinion_list = []

            for id_politico, statements in zip(ids, total_statements):

                if delta_method ==  'dynamic':
                    P = Model(statements).runlite_dynamic(l, delta, 0, self.lags_to_reckoning)
                if delta_method ==  'static':
                    P = Model(statements).runlite(l, delta)

                politician_opinion = PoliticianOpinion(id_politico, P)
                politician_opinion_list.append(politician_opinion)

            A, O, K = self.get_sets(self, politician_opinion_list)

            if self.approval_criteria(A,O):
                approval_trajectories+=1
                
            all_trajectories += 1
    
        return   (approval_trajectories/all_trajectories)
    

    def test_calculate_single_vote_probability(self, id_politico,  delta_method =  'dynamic'):

        self = self.get_post_trajectories_size_d_lags( self.lags_to_reckoning)

        list_probable_statements_after_t = self.from_politician_to_d_chopped_series[id_politico]

        list_probable_statements_after_t = list(list_probable_statements_after_t)      

        all_politician_i_statements = crop_all_statements_per_politician(self.df, id_politico)
        
        return list_probable_statements_after_t, all_politician_i_statements

    def calculate_single_vote_probability(self, id_politico,  delta_method =  'dynamic'):

        self = self.get_post_trajectories_size_d_lags( self.lags_to_reckoning)

        list_probable_statements_after_t = self.from_politician_to_d_chopped_series[id_politico]

        list_probable_statements_after_t = list(list_probable_statements_after_t)      

        all_politician_i_statements = crop_all_statements_per_politician(self.df, id_politico)

        all_trajectories = 0
        A_trajectories = 0
        O_trajectories = 0

        for statements_in_d in  list_probable_statements_after_t:

            total_statements = all_politician_i_statements + statements_in_d
    
            if delta_method ==  'dynamic':
                P = Model(total_statements).runlite_dynamic(self.l, self.delta, 0, self.lags_to_reckoning)
            if delta_method ==  'static':
                P = Model(total_statements).runlite(self.l, self.delta)

            if P == 1 : A_trajectories += 1
            if P == -1 : O_trajectories += 1

            all_trajectories += 1

        set_probability = {'A': A_trajectories/all_trajectories, 'O': O_trajectories/all_trajectories }

        return   A_trajectories, O_trajectories, all_trajectories, set_probability

    def calculate_approval_probability_by_single_vote(self, needed_votes_for_approval, delta_method =  'dynamic'):

        set_probability_by_id = {}

        n_politicians = len(self.id_politicos)

    
        for id_politico in tqdm(self.id_politicos):

            print('calculating prob for politican ' + id_politico)

            set_probability = self.calculate_single_vote_probability(id_politico, delta_method )
            set_probability_by_id[id_politico] = set_probability

        for _i in tqdm(range(needed_votes_for_approval,n_politicians)):

            print('calculating prob for politican ' + id_politico)

            prob = binomial_coefficient(n_politicians, _i)* set_probability_by_id[id_politico]['A'] * set_probability_by_id[id_politico]['O']
    
        return   prob
                

    def get_politician_trajectories(opinions_in_time: List[PoliticiansOpinionInTime], politician_id: int):
        """
        Get all different trajectories of opinions for a single politician.

        Parameters:
        - opinions_in_time: List of PoliticiansOpinionInTime instances.
        - politician_id: integer associated with politician.

        Returns:
        - A list of trajectories for the specified politician.
        """
        politician_trajectories = []

        # Iterate through the list of opinions_in_time
        for opinion_in_time in opinions_in_time:
            datetime_point = opinion_in_time.datetime

            # Find the politician's opinion at this datetime_point
            politician_opinion = next((opinion.opinion_score for opinion in opinion_in_time.politician_opinions
                                    if opinion.politician_id == politician_id), None)

            if politician_opinion is not None:
                # Append the datetime_point and opinion to the trajectories
                politician_trajectories.append((datetime_point, politician_opinion))

        return politician_trajectories
