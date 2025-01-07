
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from src.load import   (crop_statements_until_t,
                   crop_statements_from_t0_to_t, 
                   crop_statements_until_t_by_politician, 
                   crop_all_statements,crop_all_statements_per_politician)

from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
from itertools import product
from src.models import SimulateStatement, Model, PoliticianOpinion, PoliticiansOpinionInTime



import math

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


class ModelStats: 

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

        self.df = self.df[self.df['time'] < self.day_of_reckoning] # make sure that we disregard any data after the day of reckoning

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
    

    def get_votes(self, l, delta, lag,  day_of_reckoning, score = 'exp', delta_method = 'dynamic'):

        self.l = l
        self.delta = delta
        self.lag = lag

        # times = self.df.time[::lag] -- old formula

        self.get_timeframes(lag, day_of_reckoning)
    
        id_politicos = self.get_politicians()
        self.id_politicos = id_politicos

        from_time_to_politician_opinion_list = {}
        from_politician_to_opinion_history = {id_politico : [] for id_politico in id_politicos}

        for n, time_ in tqdm(enumerate(self.times)):

            politician_opinion_list = []

            for elem in crop_statements_until_t(self.df, time_): # de politico em politico

                statements, id_politico = elem

                if delta_method ==  'dynamic':
                    P = Model(statements,l, delta).runlite_dynamic( self.lags_to_reckoning - n, self.lags_to_reckoning)
                if delta_method ==  'static':
                    P = Model(statements,l, delta).runlite()

                politician_opinion = PoliticianOpinion(int(id_politico), P)
                from_politician_to_opinion_history[int(id_politico)].append(P)
                politician_opinion_list.append(politician_opinion)
                # statements, id_politico = elem

            politicians_opinion_in_time = PoliticiansOpinionInTime(politician_opinion_list, time_)
            from_time_to_politician_opinion_list[time_] = politicians_opinion_in_time

        self.from_time_to_politician_opinion_list = from_time_to_politician_opinion_list
        self.from_politician_to_opinion_history = from_politician_to_opinion_history

        return self
    

    def get_probable_vote_at_reckoning(self, statements, id_politico, l, delta,  score = 'exp', delta_method = 'dynamic'):

        if delta_method ==  'dynamic':
            P = Model(statements).runlite_dynamic(l, delta, 0, self.total_distance_from_reckoning)
        if delta_method ==  'static':
            P = Model(statements).runlite(l, delta)

        politician_opinion = PoliticianOpinion(id_politico, P)

        return politician_opinion
    
    
    def get_changes(self, silent_neutrality=None):
        """
        Counts changes of opinion in approval sets
        following model dynamic

        Args: 
        
        l - lambda parameter
        delta - delta parameter
        lag - time lag between measurement of system state

        """

        politicians_opinions_until_t = []
        total_sets = []

        changes = np.zeros(( len(self.times) , 3 ))

        A_t = []
        O_t = []
        K_t = []

        for ii, t in tqdm(enumerate(self.times)):

            politician_opinion_list = self.from_time_to_politician_opinion_list[t]
            opinions = [x.opinion for x in politician_opinion_list.politician_opinions]

            A = opinions.count(1)
            O = opinions.count(-1)
            K = opinions.count(0)  

            if silent_neutrality:
                K = K + self.deputados.NOME.count() - (A + O + K)     # silent neutrality assumption
                # K = K + self.df.NOME.count() - (A + O + K)     # silent neutrality assumption

            total_sets = total_sets + [[A,O,K]]

            A_t.append(A)
            O_t.append(O)
            K_t.append(K)

            if(ii>0):

                nA1 = int(A)
                nO1 = int(O)
                nK1 = int(K)

                changes[ii][0] = nA1 - nA
                changes[ii][1] = nO1 - nO
                changes[ii][2] = nK1 - nK

                nA = nA1
                nO = nO1
                nK = nK1

            elif(ii==0):

                nA = int(A)
                nO = int(O)
                nK = int(K)

            self.changes = changes
            self.A = A_t
            self.O = O_t
            self.K = K_t

        return self
    
    def get_sets(self, politician_opinion_list):
        """
        Get number os agents in each set

        Args: 
        
        l - lambda parameter
        delta - delta parameter
        lag - time lag between measurement of system state

        """

        opinions = [x.opinion for x in politician_opinion_list.politician_opinions]

        A = opinions.count(1)
        O = opinions.count(-1)
        K = opinions.count(0)  

        return A, O, K
    
    def approval_criteria(self,A,O):
        """
        Get aprooval as boolean

        Args: 

        A
        O
        """

        return A >= 2*O
    

    def get_fluxes(self):
        """
        CONVENÇÃO PARA OS FLUXOS
        FLUXES: A -> K ; K -> O ; A -> O; #
        """

        self.fluxes = np.zeros(( len(self.times) , 3 ))

        for ii, t in enumerate(tqdm(self.times)):

            time.sleep(1)

            if(ii>0):

                politician_opinion_list_0 = self.from_time_to_politician_opinion_list[self.times[ii-1]]
                politician_opinion_list_0 = [x.opinion for x in politician_opinion_list_0.politician_opinions]

                politician_opinion_list = self.from_time_to_politician_opinion_list[t]
                politician_opinion_list = [x.opinion for x in politician_opinion_list.politician_opinions]

                changing_opinions = pd.Series(politician_opinion_list_0) - pd.Series(politician_opinion_list[:len(politician_opinion_list_0)])
                non_zero_changing_opinions = np.where(changing_opinions != 0)[0]

                if len(non_zero_changing_opinions) > 0 :

                    po = np.array(politician_opinion_list_0)[non_zero_changing_opinions]
                    pf = np.array(politician_opinion_list)[non_zero_changing_opinions]

                    phi = np.transpose([po,pf])

                    for i in phi:

                        if (i.tolist() == [ 0. , 1.]):
                            self.fluxes[ii][0] += -1
                        if (i.tolist() == [ 1. , 0.]):
                            self.fluxes[ii][0] += 1
                        if (i.tolist() == [ 0., -1.]):
                            self.fluxes[ii][1] += 1
                        if (i.tolist() == [ -1. , 0.]):
                            self.fluxes[ii][1] += -1
                        if (i.tolist() == [ 1. ,-1.]):
                            self.fluxes[ii][2] += 1
                        if (i.tolist() == [ -1., 1.]):
                            self.fluxes[ii][2] += -1

        return self

    def get_fluxes_stats(self):
        """
        Get statistics for fluxes
        Flux Convention
        FLUXES: A -> K ; K -> O ; A -> O; 
        """
        means = [np.mean(np.transpose(self.fluxes)[0]),np.mean(np.transpose(self.fluxes)[1]),np.mean(np.transpose(self.fluxes)[2])]
        stds = [np.std(np.transpose(self.fluxes)[0]),np.std(np.transpose(self.fluxes)[1]),np.std(np.transpose(self.fluxes)[2])]

        return means, stds

    def get_P_stats(self, Plista):
        """
        Get statistics for vote set size
        
        """

        for P in Plista:
            A = np.where(P==1)
            O = np.where(P==-1)
            K = np.where( P==0 )

        means = [np.mean(np.transpose(self.fluxes)[0]),np.mean(np.transpose(self.fluxes)[1]),np.mean(np.transpose(self.fluxes)[2])]
        stds = [np.std(np.transpose(self.fluxes)[0]),np.std(np.transpose(self.fluxes)[1]),np.std(np.transpose(self.fluxes)[2])]
        
        return means, stds

    
    def study_delta(self,delta,l,lag):

        m = []
        s = []

        for d in delta:

            fluxes, Plista = self.get_fluxes_df_interval(self.df, d, l, lag)
            meanss , stdss = self.get_fluxes_stats(fluxes)
            m.append(meanss)
            s.append(stdss)
        
        return m, s

    def study_l(self, df, delta, lambd, lag):

        m = []
        s = []

        for l in lambd:

            fluxes, Plista = self.get_fluxes_df_interval(df, delta, l, lag)
            means , stds = self.get_fluxes_stats(fluxes)
            m.append(means)
            s.append(stds)
        
        return m, s

    def map_l_delta(self, df,delta,lambd,lag):

        means3d = []
        stds3d = []

        for d in delta:
            print(d)
            m , s = self.study_l(df,d,lambd,lag)
            means3d.append(m)
            stds3d.append(s)
        return means3d, stds3d

    def create_visualization(Plista,t):

        mapa = np.zeros((18,18))
        
        k = 0
        j = 0

        for i in Plista[t]:
            
            if(k%18 == 0 and k != 0):
                k = 0
                j += 1
            if i == 0:
                mapa[k][j] = 0.5
            else:
                mapa[k][j] = i
            k+=1
        
        return mapa

    def organize_politicalparty(self, t):

        df = self.deputados

        IdtoParty = {i:df[df.Id_politico == i]['Partido'].values[0] for i in df.Id_politico} 

        politician_opinions = self.from_time_to_politician_opinion_list[t]

        politician_opinion_list = [x.opinion for x in politician_opinions.politician_opinions]

        politician_id_list = [x.politician_id for x in politician_opinions.politician_opinions]

        parties = [IdtoParty[i] for i in politician_id_list]

        parties_participation = { i: parties.count(i) for i in np.unique(parties) }

        partytoopinions= {}

        for [i,j] in zip(politician_opinion_list, politician_id_list):
            #print(i,j)
            party = IdtoParty[j]
            if party in partytoopinions.keys():
                partytoopinions[party] = partytoopinions[party] + [i]
            else:
                partytoopinions[party] = [i]

        totalpartyopinion = {}

        for party in np.unique(self.deputados.Partido):

            #totalpartyopinion[party] = {1: 0, 0: + self.deputados.Partido.value_counts()[party], -1: 0}

            totalpartyopinion[party] = {1: 0, 0: 0, -1: 0}

        for party in partytoopinions.keys():
            p_A = partytoopinions[party].count(1)

           # silent neutrality assumption
           # p_K = partytoopinions[party].count(0) + self.deputados.Partido.value_counts()[party] - partytoopinions[party].count(1) - partytoopinions[party].count(-1)
            
            p_K = partytoopinions[party].count(0) 
            p_O = partytoopinions[party].count(-1)

            totalpartyopinion[party] = {1: p_A, 0: p_K, -1: p_O}

        return parties_participation, partytoopinions, totalpartyopinion

    def visualize_parties_evolution(self):  

        self.parties_opinion_evolution = []

        for t in tqdm(self.times):

            parties_participation, partytoopinions, totalpartyopinion = self.organize_politicalparty(t)

            self.parties_opinion_evolution = self.parties_opinion_evolution + [totalpartyopinion]

        return self.parties_opinion_evolution

    def serie_temporal_partido(self, party):

        serie_A =  [ self.parties_opinion_evolution[i][party][1] for i in range(len(self.parties_opinion_evolution)) ]
        serie_K =  [ self.parties_opinion_evolution[i][party][0] for i in range(len(self.parties_opinion_evolution)) ]
        serie_O =  [ self.parties_opinion_evolution[i][party][-1] for i in range(len(self.parties_opinion_evolution))]

        return serie_A, serie_K, serie_O
    
    
    def get_lists_size_m_from_list(opinion_list, m):
        """"
        Where m is posts per lag based on politician rate

        """

        n = len(opinion_list)
        n_trajetorias = round(n/m)
        sectioned_list = [ opinion_list[i * m : (i+1) * m] for i in range(n_trajetorias)] 

        return sectioned_list
    



