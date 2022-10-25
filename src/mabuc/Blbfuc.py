"""
Created on Sat Oct 22 17:31 2022
@author: lguelman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Blbfuc():
    """
    Batch Learning from Bandit Feedback with Unobserved Confounders
    
    pr_I: probability of high-income
    pr_M: probability of male
    pr_payout: probability of payout
    T: number of observations
    """
    
    def __init__(self, pr_I:float, pr_M:float, T:int, pr_payout:float=None):
        self.pr_I = pr_I
        self.pr_M = pr_M
        self.T = T 
        if pr_payout is None:
            pr_payout =  np.array([[0.25, 0.5, 0.4, 0.2],
                                   [0.5, 0.25, 0.2, 0.4]])
        self.pr_payout = pr_payout
        
        
    def sample_clients(self):
        
        C = {str(t):None for t in range(self.T)}
        I = np.random.binomial(1, self.pr_I, self.T)
        M = np.random.binomial(1, self.pr_M, self.T)
        
        for t in range(self.T):
            C[str(t)] = {'I':I[t], 'M':M[t]}
 
        self.C = C
         
        return self
    
    def play_policy(self):
        
        X = {str(t):None for t in range(self.T)}
       
        #---- Play client choice (clc), random (rct), and oracle strategies
        for t in range(self.T):
            X_clc_t = int((self.C.get(str(t))['I'] ^ self.C.get(str(t))['M']))
            # I==0 and M==0 | I==1 and M==1 ==> X=0
            # I==0 and M==1 | I==1 and M==0 == >X=1
            X_rct_t = int(np.random.binomial(1, 0.5, 1))  
            X_ora_t = int(np.array([1, 0, 0, 1])[self.C.get(str(t))['M'] + 2 * self.C.get(str(t))['I']])
            # I==0 and M==0 ==> X=1
            # I==0 and M==1 ==> X=0
            # I==1 and M==0 ==> X=0
            # I==1 and M==1 ==> X=1
            X_clc_opt_t = int(self.C.get(str(t))['M'])
            # M==0 => X==0
            # M==1 => X==1
            X_rct_opt_t = 1 - int(self.C.get(str(t))['M'])
            # M==0 ==> X==1
            # M==1 ==> X==0
            
            X[str(t)] = {'X_clc':X_clc_t, 'X_rct':X_rct_t, 'X_ora':X_ora_t,
                         'X_clc_opt':X_clc_opt_t, 'X_rct_opt':X_rct_opt_t}
  
        self.X = X

        return self    
    
    
    def get_payout_t(self, X, I, M):
        
        if X==0:
            Y = int(np.random.binomial(1, self.pr_payout[0,M + 2*I], 1))
        else: 
            Y = int(np.random.binomial(1, self.pr_payout[1,M + 2*I], 1))
            
        return Y
    
    #X==0
    #I==0 and M==0 ==> 0.25
    #I==0 and M==1 ==> 0.50
    #I==1 and M==0 ==> 0.40
    #I==1 and M==1 ==> 0.20
           
    
    def get_payout(self):
        
        Y = {str(t):None for t in range(self.T)}
        
        #Y = {'Y'+k[1:5]:dgp.get_payout(v, C.get('D'), C.get('B')) for (k,v) in X.items()}
    
        for t in range(self.T):
            Y[str(t)] = {'Y_clc': self.get_payout_t(self.X.get(str(t))['X_clc'], self.C.get(str(t))['I'], self.C.get(str(t))['M']),
                         'Y_rct': self.get_payout_t(self.X.get(str(t))['X_rct'], self.C.get(str(t))['I'], self.C.get(str(t))['M']),
                         'Y_ora': self.get_payout_t(self.X.get(str(t))['X_ora'], self.C.get(str(t))['I'], self.C.get(str(t))['M']),
                         'Y_clc_opt': self.get_payout_t(self.X.get(str(t))['X_clc_opt'], self.C.get(str(t))['I'], self.C.get(str(t))['M']),
                         'Y_rct_opt': self.get_payout_t(self.X.get(str(t))['X_rct_opt'], self.C.get(str(t))['I'], self.C.get(str(t))['M']),
                        
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
            dgp = Blbfuc(pr_I=self.pr_I, pr_M=self.pr_M, T=self.T)
            dgp.sample_clients()
            dgp.play_policy()
            dgp.get_payout()
        
            for t in range(self.T):
                data_r[str(t)] = {**dgp.Y[str(t)],**dgp.X[str(t)], **dgp.C[str(t)]}
         
            df_r = pd.DataFrame.from_dict(data_r, orient='index')
       
            data.append(df_r)
        
        self.data = data 
        
        return self

    
    def get_counterfactuals(self):
        
        data = self.data
        R = len(data)
        
        for r in range(R):
            
            Py_dox0_m0, Py_dox0_m1, Py_dox1_m0, Py_dox1_m1 =  \
                data[r].groupby(['X_rct', 'M'])['Y_rct'].agg('mean').values
            
            Py_x0_m0, Py_x0_m1, Py_x1_m0, Py_x1_m1 =  \
                data[r].groupby(['X_clc', 'M'])['Y_clc'].agg('mean').values
                
            Px1_m0, Px1_m1  = \
                data[r].groupby(['M'])['X_clc'].agg('mean').values
            
            Px0_m0 = 1-Px1_m0
            Px0_m1 = 1-Px1_m1
            
            Pyx1_x0_m1 = \
                (Py_dox1_m1 - Py_x1_m1 * Px1_m1) * 1/Px0_m1
                
            Pyx1_x0_m0 = \
                (Py_dox1_m0 - Py_x1_m0 * Px1_m0) * 1/Px0_m0
                  
            Pyx0_x1_m1 = \
                (Py_dox0_m1 - Py_x0_m1 * Px0_m1) * 1/Px1_m1
                
            Pyx0_x1_m0 = \
                (Py_dox0_m0 - Py_x0_m0 * Px0_m0) * 1/Px1_m0
                
            
            def _action_map(df):
            
                M = df['M']
                X_clc = df['X_clc']
            
                if M==1 and X_clc==0:
                    if Pyx1_x0_m1 - Py_x0_m1 > 0: # same for all obs within data[r]
                        X_csl = 1
                    else:
                        X_csl = 0
                    
                elif M==1 and X_clc==1:
                    if Py_x1_m1 - Pyx0_x1_m1 > 0:
                        X_csl = 1
                    else:
                        X_csl = 0
                
                elif M==0 and X_clc==0:
                    if Pyx1_x0_m0 - Py_x0_m0 > 0:
                        X_csl = 1
                    else:
                        X_csl = 0
                
                elif M==0 and X_clc==1:
                    if Py_x1_m0 - Pyx0_x1_m0 > 0:
                        X_csl = 1
                    else:
                        X_csl = 0
                        
                return  X_csl 
                    
           
            data[r]['X_csl'] = data[r].apply(_action_map, axis=1)   
            
            Y_csl = []
            for t in range(self.T):
                Y_csl.append(self.get_payout_t(X=data[r]['X_csl'].iloc[t], 
                                  I = data[r]['I'].iloc[t],
                                  M = data[r]['M'].iloc[t]))
                
            data[r]['Y_csl'] = Y_csl
          
        self.data = data
            
        return self
            
                
    def get_stats(self):
        
        data = self.data
        R = len(data)
       
        clc_summary = np.zeros(shape=[R, 4])
        rct_summary = np.zeros(shape=[R, 4])
        ora_summary = np.zeros(shape=[R, 4])
        csl_summary = np.zeros(shape=[R, 4])
      
        
        for r in range(R):
            clc_summary[r] = data[r].groupby(['X_clc', 'M'])['Y_clc'].agg('mean').values
            rct_summary[r] = data[r].groupby(['X_rct', 'M'])['Y_rct'].agg('mean').values
            ora_summary[r] = data[r].groupby(['X_ora', 'M'])['Y_ora'].agg('mean').values
            #csl_summary[r] = data[r].groupby(['X_csl', 'M'])['Y_csl'].agg('mean').values
            
        
        with np.printoptions(precision=3, suppress=True):
            
            print("\n Oracle Payout [P-F, P-M, D-F, D-M]: \n")
            print("Mean:", np.mean(ora_summary, axis=0))
            print("S.E.:", np.std(ora_summary, ddof=1, axis=0) / np.sqrt(R))
       
            print("\n Client Choice Payout [P-F, P-M, D-F, D-M]: \n")
            print("Mean:", np.mean(clc_summary, axis=0))
            print("S.E.:", np.std(clc_summary, ddof=1, axis=0) / np.sqrt(R))
        
            print("\n RCT Payout [P-F, P-M, D-F, D-M]: \n")
            print("Mean:", np.mean(rct_summary, axis=0))
            print("S.E.:", np.std(rct_summary, ddof=1, axis=0) / np.sqrt(R))
            
            #print("\n RCT Payout [P-F, P-M, D-F, D-M]: \n")
            #print("Mean:", np.mean(csl_summary, axis=0))
            #print("S.E.:", np.std(csl_summary, ddof=1, axis=0) / np.sqrt(R))
            
        
        ymean_ora = np.zeros(shape=[R])
        ymean_clc = np.zeros(shape=[R])
        ymean_rct = np.zeros(shape=[R])
        ymean_clc_opt = np.zeros(shape=[R])
        ymean_rct_opt = np.zeros(shape=[R])
        ymean_csl_opt = np.zeros(shape=[R])
        
        for r in range(R):
            ymean_ora[r] = data[r]['Y_ora'].agg('mean')
            ymean_clc[r] = data[r]['Y_clc'].agg('mean')
            ymean_rct[r] = data[r]['Y_rct'].agg('mean')
            ymean_clc_opt[r] = data[r]['Y_clc_opt'].agg('mean')
            ymean_rct_opt[r] = data[r]['Y_rct_opt'].agg('mean')
            ymean_csl_opt[r] = data[r]['Y_csl'].agg('mean')
        
        
        with np.printoptions(precision=3, suppress=True):
            
            print("\n Oracle Payout \n")
            print("Mean:", np.mean(ymean_ora))
            print("S.E.:", np.std(ymean_ora, ddof=1) / np.sqrt(R))
            
            print("\n Client choice Payout \n")
            print("Mean:", np.mean(ymean_clc))
            print("S.E.:", np.std(ymean_clc, ddof=1) / np.sqrt(R))
            
            print("\n RCT Payout \n")
            print("Mean:", np.mean(ymean_rct))
            print("S.E.:", np.std(ymean_rct, ddof=1) / np.sqrt(R))
            
            print("\n Client choice Optimal Payout \n")
            print("Mean:", np.mean(ymean_clc_opt))
            print("S.E.:", np.std(ymean_clc_opt, ddof=1) / np.sqrt(R))
            
            print("\n RCT Optimal Payout \n")
            print("Mean:", np.mean(ymean_rct_opt))
            print("S.E.:", np.std(ymean_rct_opt, ddof=1) / np.sqrt(R))
       
            print("\n CSL Optimal Payout \n")
            print("Mean:", np.mean(ymean_csl_opt))
            print("S.E.:", np.std(ymean_csl_opt, ddof=1) / np.sqrt(R))
   
        
        self.clc_summary = clc_summary
        self.rct_summary = rct_summary
        self.ora_summary = ora_summary
        self.ymean_ora = ymean_ora
        self.ymean_clc = ymean_clc
        self.ymean_rct = ymean_rct
        self.ymean_clc_opt = ymean_clc_opt
        self.ymean_rct_opt = ymean_rct_opt
        
        return self
        
    
    
        
        
###

m = Blbfuc(pr_I=0.5, pr_M=0.5, T=1000)
m.generate_samples(R=100)
m.get_counterfactuals()
m.get_stats()

        

