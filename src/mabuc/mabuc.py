"""
Created on Wed Oct  5 07:37:31 2022
@author: lguelman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Mabuc():
    """
    Data Generating Process 
    
    pr_D: probability of a gambler being drunk
    pr_B: probability that machine would be blinking
    pr_payout: probability of payout
    T: number of trials
    """
    
    def __init__(self, pr_D:float, pr_B:float, T:int, pr_payout:float=None):
        self.pr_D = pr_D
        self.pr_B = pr_B
        self.T = T 
        if pr_payout is None:
            pr_payout =  np.array([[0.1, 0.5, 0.4, 0.2],
                                   [0.5, 0.1, 0.2, 0.4]])
        self.pr_payout = pr_payout
        
        
    def sample_players(self):
        
        C = {str(t):None for t in range(self.T)}
        D = np.random.binomial(1, self.pr_D, self.T)
        B = np.random.binomial(1, self.pr_B, self.T)
        
        for t in range(self.T):
            C[str(t)] = {'D':D[t], 'B':B[t]}
 
        self.C = C
         
        return self
    
    def pull_arm(self):
        
        X = {str(t):None for t in range(self.T)}
        
        a0, b0, a1, b1 = (np.ones(4))
        s0, f0, s1, f1 = (np.ones(4))
        
        # seed distribution
        p_Y_X = [self.pr_payout[0][0] * (1- self.pr_D) + self.pr_payout[0][3] * self.pr_D,
                 self.pr_payout[1][1] * (1- self.pr_D) + self.pr_payout[1][2] * self.pr_D]
        
        P_X = [(1-self.pr_D), self.pr_D]
        
        # initialize est. payout for counter-intuition 
        Q1 = [float(np.random.beta(s0, f0, size =1)),  
              float(np.random.beta(s1, f1, size =1))]
                 
           
        for t in range(self.T):

            #---- Play intuition, random, and optimal strategies
            X_int_t = int((self.C.get(str(t))['D'] ^ self.C.get(str(t))['B']))
            X_rnd_t = int(np.random.binomial(1, 0.5, 1))  
            X_opt_t = int(np.array([1, 0, 0, 1])[self.C.get(str(t))['B'] + 2 * self.C.get(str(t))['D']])
            
            #---- Play Thompson Sampling 
            theta0 = float(np.random.beta(a0, b0, size =1))
            theta1 = float(np.random.beta(a1, b1, size =1))
            X_ts_t = np.argmax([theta0, theta1])
            Y_ts_t = self.get_payout_t(X_ts_t, self.C.get(str(t))['D'], self.C.get(str(t))['B'])
           
            if X_ts_t == 0:
                a0+= Y_ts_t
                b0+= 1 - Y_ts_t
            else:
                a1+= Y_ts_t
                b1+= 1 - Y_ts_t
                
            #---- Play Causal Thompson Sampling
            
            # counter-intuition
            if X_int_t == 1:
                X_cint_t = 0
            else:
                X_cint_t=1 
                
            # est. payout for counter-intuition    
            Q1_t = Q1[X_cint_t]
            
            # est. payout for intuition
            Q2_t = p_Y_X[X_int_t]
            w = np.ones(2)
            bias = float(1 - np.abs([Q1_t-Q2_t]))
            if Q1_t > Q2_t:
                w[X_int_t] = bias
            else:
                w[X_cint_t] = bias
            
            X_cts_t = np.argmax([float(np.random.beta(s0, f0, size =1))*w[0],
                                 float(np.random.beta(s1, f1, size =1))*w[1]])
            Y_cts_t = self.get_payout_t(X_cts_t, self.C.get(str(t))['D'], self.C.get(str(t))['B'])
           
            # update counts 
            if X_cts_t == 0:
                s0+= Y_cts_t
                f0+= 1 - Y_cts_t 
               
            else:
                s1+= Y_cts_t
                f1+= 1 - Y_cts_t
               
            # update Q1
            Q1 = [1/P_X[1] * ( s0/(s0+f0) -  p_Y_X[0] * P_X[0] ), 
                  1/P_X[0] * ( s1/(s1+f1) -  p_Y_X[1] * P_X[1] )
                  ]
                
            X[str(t)] = {'X_int':X_int_t, 'X_rnd':X_rnd_t, 'X_opt':X_opt_t, 
                         'X_ts':X_ts_t, 'X_cts':X_cts_t}
         
        self.X = X
    
        return self    
    
    def get_payout_t(self, X, D, B):
        
        if X==0:
            Y = int(np.random.binomial(1, self.pr_payout[0,B + 2*D], 1))
        else: 
            Y = int(np.random.binomial(1, self.pr_payout[1,B + 2*D], 1))
            
        return Y
    
    def get_payout(self):
        
        Y = {str(t):None for t in range(self.T)}
        
        #Y = {'Y'+k[1:5]:dgp.get_payout(v, C.get('D'), C.get('B')) for (k,v) in X.items()}
    
        for t in range(self.T):
            Y[str(t)] = {'Y_int': self.get_payout_t(self.X.get(str(t))['X_int'], self.C.get(str(t))['D'], self.C.get(str(t))['B']),
                         'Y_rnd': self.get_payout_t(self.X.get(str(t))['X_rnd'], self.C.get(str(t))['D'], self.C.get(str(t))['B']),
                         'Y_opt': self.get_payout_t(self.X.get(str(t))['X_opt'], self.C.get(str(t))['D'], self.C.get(str(t))['B']),
                         'Y_ts':  self.get_payout_t(self.X.get(str(t))['X_ts'], self.C.get(str(t))['D'], self.C.get(str(t))['B']),
                         'Y_cts': self.get_payout_t(self.X.get(str(t))['X_cts'], self.C.get(str(t))['D'], self.C.get(str(t))['B']),
                         }
         
        self.Y = Y
        
        return self
    
    def generate_samples(self, R:int=100):
        
        """
        R: Number of repetitions of simulation
        """
        
        data = []
        
        for _ in range(R):
            
            data_r = {str(t):None for t in range(self.T)}
            dgp = Mabuc(pr_D=self.pr_D, pr_B=self.pr_B, T=self.T)
            dgp.sample_players()
            dgp.pull_arm()
            dgp.get_payout()
        
            for t in range(self.T):
                data_r[str(t)] = {**dgp.Y[str(t)],**dgp.X[str(t)], **dgp.C[str(t)]}
         
            df_r = pd.DataFrame.from_dict(data_r, orient='index')
       
            data.append(df_r)
        
        self.data = data 
        
        return self
        
                
    def get_stats(self):
        
        data = self.data
        R = len(data)
       
        int_summary = np.zeros(shape=[R, 2])
        rnd_summary = np.zeros(shape=[R, 2])
        opt_summary = np.zeros(shape=[R, 2])
        ts_summary =  np.zeros(shape=[R, 2])
        cts_summary = np.zeros(shape=[R, 2])
        
        for r in range(R):
            int_summary[r] = data[r].groupby(['X_int'])['Y_int'].agg('mean').values
            rnd_summary[r] = data[r].groupby(['X_rnd'])['Y_rnd'].agg('mean').values
            opt_summary[r] = data[r].groupby(['X_opt'])['Y_opt'].agg('mean').values
            ts_summary[r]  = data[r].groupby(['X_ts'])['Y_ts'].agg('mean').values
            cts_summary[r] = data[r].groupby(['X_cts'])['Y_cts'].agg('mean').values
        
        with np.printoptions(precision=3, suppress=True):
            
            print("\n Optimal Payout [arm1, arm2]: \n")
            print("Mean:", np.mean(opt_summary, axis=0))
            print("S.E.:", np.std(opt_summary, ddof=1, axis=0) / np.sqrt(R))
       
            print("\n Intuition Payout [arm1, arm2]: \n")
            print("Mean:", np.mean(int_summary, axis=0))
            print("S.E.:", np.std(int_summary, ddof=1, axis=0) / np.sqrt(R))
        
            print("\n Random Payout [arm1, arm2]: \n")
            print("Mean:", np.mean(rnd_summary, axis=0))
            print("S.E.:", np.std(rnd_summary, ddof=1, axis=0) / np.sqrt(R))
        
            print("\n Thompson Sampling Payout [arm1, arm2]: \n")
            print("Mean:", np.mean(ts_summary, axis=0))
            print("S.E.:", np.std(ts_summary, ddof=1, axis=0) / np.sqrt(R))
       
            print("\n Causal Thompson Sampling Payout [arm1, arm2]: \n")
            print("Mean:", np.mean(cts_summary, axis=0))
            print("S.E.:", np.std(cts_summary, ddof=1, axis=0) / np.sqrt(R))
            
        self.int_summary = int_summary
        self.rnd_summary = rnd_summary
        self.opt_summary = opt_summary
        self.ts_summary  = ts_summary
        self.cts_summary = cts_summary
       
        return self
        
    
    def plot(self):
        
        data = self.data
        R = len(data)
        
        int_opt = np.zeros(shape=[self.T, R])
        rnd_opt=  np.zeros(shape=[self.T, R])
        ts_opt =  np.zeros(shape=[self.T, R])
        cts_opt = np.zeros(shape=[self.T, R])
        
        # played optimal action?
        for r in range(R):
            int_opt[:,r] = data[r]['X_int'] == data[r]['X_opt']
            rnd_opt[:,r] = data[r]['X_rnd'] == data[r]['X_opt']
            ts_opt[:,r] = data[r]['X_ts'] == data[r]['X_opt']
            cts_opt[:,r] = data[r]['X_cts'] == data[r]['X_opt']
         
        int_opt_summary = np.mean(int_opt, axis=1)
        rnd_opt_summary = np.mean(rnd_opt, axis=1)
        ts_opt_summary = np.mean(ts_opt, axis=1)
        cts_opt_summary = np.mean(cts_opt, axis=1)
        
        cumsum_ones = np.cumsum(np.ones(self.T))
        
        plt.plot(cumsum_ones, int_opt_summary, label ="Intuition")
        plt.plot(cumsum_ones, rnd_opt_summary, label ="Random")
        plt.plot(cumsum_ones, ts_opt_summary, label ="TS")
        plt.plot(cumsum_ones, cts_opt_summary, label ="CTS")
        plt.title('Probability of Optimal Action')
        plt.xlabel('Trial')
        plt.ylabel('Probability')
        plt.legend()
        plt.ylim((0,1))
        plt.show()
        
        

