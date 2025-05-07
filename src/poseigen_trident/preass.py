import numpy as np

import torch
import torch.nn as nn

import poseigen_seaside.basics as se

import poseigen_trident.utils as tu
import poseigen_trident.prongs as tp


class SimpNet(nn.Module):
    def __init__(self, 
                 dim_i = (8,1,1), dim_f = (1,1,1),                  
                 P1_mods = 3, P1_cf_i = 50, P1_cf_ns = 1, P1_cf_pu = 10,
                 P1_dropout = None, 
                 batchnorm = 'before', bias = True,
                 activations = nn.ReLU(), activation_f = None): 
        
        super(SimpNet, self).__init__()
        
        self.P1 = nn.Sequential(* tp.Prong_X(dim_i, dim_f, mods = P1_mods, 
                                          cf_i = P1_cf_i, cf_ns = P1_cf_ns, cf_pu = P1_cf_pu,
                                          dropout = P1_dropout, activations = activations,
                                          batchnorm = batchnorm, bias = bias,
                                          out = True, activation_f = activation_f))                       
                                
    def forward(self,x):
        x = self.P1(x)
        return x
    
    SimpNetDict = {'dim_i': [[(8,1,1)],'cat'], #Set as a single dim_i
                   'dim_f': [[(1,1,1)],'cat'], #Set as a single dim_f
                   
                   'P1_mods': [[1,5], 'int'],
                   'P1_cf_i': [[5,50],'int'],
                   'P1_cf_ns': [[1, 1.6], 'cat'],
                   'P1_dropout': [[0,0.5], 'float'],
                   
                   'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
                   'activation_f': [[None], 'cat']}
    



class  StandardNN(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 mid_f = 200, mid_l = 10, 

                 P1_mods = 3, P1_mods_ns = 0.3, 
                 P1_ck_base = 5, P1_ck_i = 13, 
                 P1_pool_func = nn.MaxPool2d, P1_pool_k2s = 1,

                 P2_mods = 3,
                 P2_cf_pu_m = 1, P2_dropout = 0.2, 

                 cf_ns = 1, #####
                 activations = nn.ReLU(), activation_f = None,
                 batchnorm = 'before'):
        
        super(StandardNN, self).__init__()

        dim_mid = (mid_f, mid_l, 1)

        P1_args = {'dim_i': dim_i, 'dim_f': dim_mid,
                   'mods': P1_mods, 'mods_ns': P1_mods_ns,
                   'cf_ns': cf_ns,
                   'ck_base': P1_ck_base, 'ck_i': P1_ck_i,
                   'pool_k2s': P1_pool_k2s, 'pool_func': P1_pool_func,
                   'activations': activations, 'batchnorm': batchnorm}
        
        self.P1 = nn.Sequential(*tp.Prong_Y(**P1_args))

        P2_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                   'mods': P2_mods,
                   'cf_ns': cf_ns, 'cf_pu': P2_cf_pu_m * mid_f,
                   'activations': activations, 'activation_f': activation_f,
                   'batchnorm': batchnorm,
                   'dropout': P2_dropout, 'out': True}
        
        self.P2 = nn.Sequential(*tp.Prong_X(**P2_args))
        
    def forward(self, x):

        x = self.P1(x)
        x = self.P2(x)

        return x
    


stand_args = {'dim_i': (1,249,4), 
              'dim_f': (1,1,1),
              'mid_f': 200, 'mid_l': 10, 
              
              'P1_mods': 3, 'P1_mods_ns': 0.3, 
              'P1_ck_base': 5, 'P1_ck_i': 13, 
              'P1_pool_func': nn.MaxPool2d, 'P1_pool_k2s': 1,
              
              'P2_mods': 3,'P2_cf_pu_m': 1, 'P2_dropout': 0.2, 

              'cf_ns': 1,
              
              'activations': nn.ReLU(), 'activation_f': None, 'batchnorm': 'before'}

stand_dict = {'dim_i': [[(1,249,4)],'cat'],
              'dim_f': [[(2,1,1)],'cat'],
              'mid_f': [[50, 150, 250, 350, 450],'cat'],
              'mid_l': [[1, 15, 25, 50], 'cat'], 
              
              'P1_mods': [[2, 4, 6], 'cat'], 
              'P1_mods_ns': [[0.3, 0.5], 'cat'], 
              'P1_ck_base': [[1, 5, 10], 'cat'],
              'P1_ck_i': [[11, 14, 17, 20], 'cat'],
              'P1_pool_func': [[nn.MaxPool2d],'cat'],
              'P1_pool_k2s': [[1],'cat'],
              
              'P2_mods': [[1, 2, 3], 'cat'],
              'P2_cf_pu_m': [[0.5, 1, 2], 'cat'], 
              'P2_dropout': [[0, 0.1, 0.2], 'cat'], 

              'cf_ns': [[1, 1.1, 1.2], 'cat'],              
              'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
              'activation_f': [[None], 'cat'],
              'batchnorm': [[None, 'before'], 'cat']}






