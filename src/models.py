
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from src.load import crop_statements_until_t
from dataclasses import dataclass
from datetime import datetime

#PLista deve ser substituido por lista de objetos PoliticiansOpinionInTime

@dataclass
class PoliticianOpinion:
    """Class for identifying a single politician opinion"""
    politician_id: int
    opinion: int 

@dataclass
class PoliticiansOpinionInTime:
    """Class for keeping track of politician opinion evolution"""
    politician_opinions: list[PoliticianOpinion]
    datetime: datetime


class SimulateStatement:

    def __init__(self, N, maxtweets):
        self.N = N
        self.maxtweets = maxtweets
    
    def np_continuous(self):
        """
        cria tweets um vetor com NxMaxTweets
        statements que podem assumir valor continuous
        """
        statements = np.zeros((self.N,self.maxtweets))
        for i in range(0,self.N):
            statements[i] =  np.random.uniform(-1,1,self.maxtweets)

        return statements
    
    def np_binary(self):
        """
        cria tweets um vetor com NxMaxTweets
        statements que podem assumir valor -1 ou 1
        """
        statements = np.zeros((self.N,self.maxtweets))
        for i in range(0,self.N):
            #statements[i] =  np.random.uniform(-1,1,T)
            statements[i] = np.random.randint(0,2,self.maxtweets)
            statements[i][np.where(statements[i]==0)]=-1
        return statements


    def list_continuous(self):
        """
        cria tweets um vetor com NxArbitrario (tamanho = # posts do politico)
        statements que podem assumir valor continuous
        """
        statements = []
        for i in range(0,self.N):
            maxt = np.random.randint(0,self.maxtweets)
            statementsi =  np.random.uniform(-1,1,maxt)
            statements.append(statementsi)
        return statements

    # cria tweets um vetor com NxArbitrario (tamanho = # posts do politico)
    # statements que podem assumir valor -1 ou 1

    def list_binary(self):
        """
        cria tweets um vetor com NxArbitrario (tamanho = # posts do politico)
        statements que podem assumir valor -1 ou 1
        """
        statements = []
        for i in range(0,self.N):
            maxt = np.random.randint(0,self.maxtweets)
            statementsi =  np.random.randint(0,2,maxt)
            statementsi[np.where(statementsi==0)]=-1
            statements.append(statementsi)

        return statements

 
class Model: 
    
    def __init__(self, tau, lambd, delta):
        self.N = len(tau)
        self.tau = tau
        self.l = lambd
        self.delta= delta

    def lastOr0(obj):

        if len(obj)==0:
            return 0
        else:
            return obj[-1]

    def h_exp(self):

        h = np.zeros(self.N)

        h[0] = self.tau[0]

        for i in range(1,self.N):

            h[i] = self.l * h[i-1] + (1-self.l) * self.tau[i]

        return h

    # End result Score

    def h_exp_escalar(self):

        h =  self.tau[0]

        for i in range(1,self.N):

            h = self.l * h + (1-self.l) * self.tau[i]

        return h

    # Score as mean of posts

    def h_mean(self):
        return [np.mean(self.tau[:i]) for i in range(len(self.tau))]


    def classifier(self, scores):
        h=[]
        for i in range(len(scores)):
            obj = scores[i]
            if obj<-self.delta:
                h.append(-1)
            if obj>self.delta:
                h.append(1)
            if obj<self.delta and obj>-self.delta:
                h.append(0)

        return h
    
    def dynamic_classifier(self, scores, distance_from_reckoning, time_of_reckoning):
        
        h=[]

        delta = (distance_from_reckoning/time_of_reckoning)*self.delta

        for i in range(len(scores)):
            obj = scores[i]
            if obj<-self.delta:
                h.append(-1)
            if obj>self.delta:
                h.append(1)
            if obj<self.delta and obj>-self.delta:
                h.append(0)

        return h

    def run(self, method='exp'): # t é n de enesimo tweet

        if method=='exp':

            function = self.h_exp
            scores = function(self.l)

        if method=='mean':
            function = self.h_mean
            scores = function()

        return self.classifier(scores)

    def classifierlite(self,score):

        if score<-self.delta: return -1
        if score>self.delta: return 1
        if score<self.delta and score>-self.delta: return 0

    def classifierlite_dynamic(self, score, current_distance_from_reckoning, starting_distance_from_reckoning):

        delta = (current_distance_from_reckoning/starting_distance_from_reckoning)*self.delta

        if score<=-delta: return -1
        if score>delta: return 1
        if score<delta and score>-delta: return 0
 

    def runlite(self,  method='exp'): # t é n de enesimo tweet

        if method=='exp':
            function = self.h_exp_escalar
            scores = function()

        if method=='mean':
            function = self.h_mean
            scores = function()

        return self.classifierlite(scores,self.delta)
    
    def runlite_dynamic(self, current_distance_from_reckoning, total_distance_from_reckoning, method='exp'): # t é n de enesimo tweet

        if method=='exp':
            function = self.h_exp_escalar
            scores = function()

        if method=='mean':
            function = self.h_mean
            scores = function()

        return self.classifierlite_dynamic(scores, current_distance_from_reckoning, total_distance_from_reckoning)
    
    def runfull(self, method='exp'): # t é n de enesimo tweet

        if method=='exp':
            function = self.h_exp_escalar
            scores = function()

        if method=='mean':
            function = self.h_mean
            scores = function()

        return self.classifierlite(scores)
