from math import *
import numpy as np
from copy import deepcopy


def train_RBM(RBM, epochs, learning_rate, batch_size, data):
    W, a, b = RBM.W, RBM.a, RBM.b
    number_batches = ceil(len(data) / batch_size)
    x = deepcopy(data)
    accuracies = []
    for epoch in range(1, epochs + 1) :
        np.random.shuffle(x)
        for n_batch in range(number_batches) :
            begin = n_batch * batch_size
            end = min((n_batch+1) * batch_size, len(x))
            batch_data =  x[begin:end]
            current_batch_size = end - begin 
            grad_w, grad_a, grad_b = 0, 0, 0
        
            v_0 = batch_data
            ph0 = RBM.entree_sortie_RBM(v_0)
            h_0 = (np.random.random(ph0.shape) < ph0) * 1
            pv1 = RBM.sortie_entree_RBM(h_0)
            v_1 = (np.random.random(pv1.shape) < pv1) * 1
            ph1 = RBM.entree_sortie_RBM(v_1)
           
            grad_w = v_0.T@ph0 - v_1.T@ph1
            grad_a = np.sum(v_0 - v_1, axis=0)
            grad_b = np.sum(ph0 - ph1, axis=0)
            grad_w, grad_a, grad_b = grad_w/current_batch_size, grad_a/current_batch_size, grad_b/current_batch_size
         
            W += learning_rate * grad_w
            a += learning_rate * grad_a
            b += learning_rate * grad_b
        
        h_predict = RBM.entree_sortie_RBM(x)
        x_predict = RBM.sortie_entree_RBM(h_predict)
        error = np.sum((x-x_predict)**2)
        print("epoch ", epoch)
        print("Error : ", error)
        print("Error per image :", error/x.shape[0])
        print("Error per pixel", error/(x.shape[0]*x.shape[1]))
        accuracies.append(1-error/(x.shape[0]*x.shape[1]))
    return RBM, accuracies

