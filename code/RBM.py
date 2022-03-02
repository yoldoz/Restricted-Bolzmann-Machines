import numpy as np 
class RBM : 
    def __init__(self,p,q ) :
        self.b = np.zeros(q)
        self.a = np.zeros(p)
        self.W= np.random.randn(p, q)
        
        
        
    def entree_sortie_RBM(self,data_v):
        Z = np.dot(data_v,self.W)+self.b 
        data_h = 1/(1 + np.exp(-Z)) 
        return data_h
    
    
        
    def sortie_entree_RBM(self,data_h) :
        Z= np.dot(data_h,self.W.T)+self.a
        data_v = 1/(1 + np.exp(-Z)) 
        
        return data_v

