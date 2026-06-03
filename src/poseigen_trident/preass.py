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



class DivoNet(nn.Module):
    
    def __init__(self, 
                 dim_i = (8,1,1),
                 two_outs = False, 
                 comb = 'sub', 
                 P1_mods = 3, P1_cf_i = 50, P1_cf_ns = 1, P1_dropout = None, 
                 P1_cf_pu = 10,
                 batchnorm = 'before', bias = True,
                 activations = nn.ReLU(), activation_f = None): 
        
        super(DivoNet, self).__init__()

        #DivoNet always outputs a SINGLE VALUE 

        outies = 2 if two_outs else 1
        if two_outs is False: comb = None

        prong_out = (outies,1,1)
        
        self.P1 = nn.Sequential(* tp.Prong_X(dim_i, prong_out, mods = P1_mods, 
                                          cf_i = P1_cf_i, cf_ns = P1_cf_ns, cf_pu = P1_cf_pu,
                                          dropout = P1_dropout, activations = activations,
                                          batchnorm = batchnorm, bias = bias,
                                          out = True, activation_f = None))      

        if comb is not None: 
            if comb == 'div': self.Comb = tu.DivideLayer()
            if comb == 'sub': self.Comb = tu.SubtractLayer()
        else: self.Comb = nn.Identity()

        self.actf = nn.Identity() if activation_f is None else activation_f

                                
    def forward(self,x):
        x = self.P1(x)
        x = self.Comb(x)
        x = self.actf(x)

        if torch.any(torch.isnan(x)):
            print('OUTPUTTING NAN, WE GOT A PROBLEM HERE.')

        return x
    
    DivoNetDict = {'dim_i': [[(8,1,1)],'cat'], #Set as a single dim_i
                   
                   'two_outs': [[True, False], 'cat'],
                   'comb': [[True, False], 'cat'],
                   
                   'P1_mods': [[1,5], 'int'],
                   'P1_cf_i': [[5,50],'int'],
                   'P1_cf_ns': [[1, 1.6], 'cat'],
                   'P1_dropout': [[0,0.5], 'float'],
                   
                   'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
                   'activation_f': [[None], 'cat']}
    

#=====================================

class  RegularNN(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 mid_f = 200,

                 P1_cf_i = None,
                 P1_ck_i = 13, P1_ck_base = 5,
                 P1_pool_func = nn.MaxPool2d,
                 P1_pk_base = 2, P1_pk_s = 2, 

                 P2_mods = 3,
                 P2_cf_pu_m = 1, P2_dropout = 0.2, 

                 cf_ns = 1, #####
                 activations = nn.ReLU(), activation_f = None,
                 batchnorm = 'before'):
        
        super(RegularNN, self).__init__()

        dim_mid = (mid_f, 1, 1)

        P1_args = {'dim_i': dim_i, 'dim_f': dim_mid,
                   
                   'cf_ns': cf_ns, 'cf_i': P1_cf_i, 
                   'ck_i': P1_ck_i, 'ck_base': P1_ck_base,

                   'pool_func': P1_pool_func, 
                   'pk_base': P1_pk_base, 'pk_s': P1_pk_s, 
                   'activations': activations, 'batchnorm': batchnorm}
        
        self.P1 = nn.Sequential(*tp.Prong_W(**P1_args))

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
    

reg_args = {'dim_i': (1,249,4), 
              'dim_f': (1,1,1),
              'mid_f': 200,
              
              'P1_cf_i': None,
              'P1_ck_i': 13, 'P1_ck_base': 5,
              'P1_pool_func': nn.MaxPool2d,
              'P1_pk_base': 2, 'P1_pk_s': 2, 

              'P2_mods': 3,'P2_cf_pu_m': 1, 'P2_dropout': 0.2, 

              'cf_ns': 1,
            
              'activations': nn.ReLU(), 'activation_f': None, 'batchnorm': 'before'}

reg_dict = {'dim_i': [[(1,249,4)],'cat'],
              'dim_f': [[(1,1,1)],'cat'],
              'mid_f': [[150, 250, 350, 450],'cat'],
              
              'P1_cf_i': [[25, 50, 100], 'cat'],
              'P1_ck_i': [[8, 11, 14, 17, 20], 'cat'],
              'P1_ck_base': [[5, 7, 9], 'cat'],
              
              'P1_pool_func': [[nn.MaxPool2d, nn.AvgPool2d],'cat'],
              'P1_pk_base': [[0, 2, 4], 'cat'],
              'P1_pk_s': [[2, 3, 4], 'cat'],

              
              'P2_mods': [[1, 2, 3], 'cat'],
              'P2_cf_pu_m': [[0.5, 1, 2], 'cat'], 
              'P2_dropout': [[0, 0.1, 0.2], 'cat'], 

              'cf_ns': [[1, 1.1, 1.2], 'cat'],              
              'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
              'activation_f': [[None], 'cat'],
              'batchnorm': [[None, 'before'], 'cat']}


def Reset_RegularNN(regularnn):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 

    lconvs, ldens = len(regularnn.P1), len(regularnn.P2)

    for i in np.arange(lconvs): 
        if isinstance(regularnn.P1[i], nn.Conv2d): 
            regularnn.P1[i].reset_parameters()
    
    for i in np.arange(ldens): 
        if isinstance(regularnn.P2[i], nn.Conv2d): 
            regularnn.P2[i].reset_parameters()

    return regularnn


#=====================================

'''Champion: Applies CPMs for sequence embedding'''

class ChampionNN(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 

                 kE_k = 15, kE_cf_m = 100, kE_cf_ns = 1, 
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 
                 kE_freeze = False,

                 PR_cf_m = 2, PR_ck_base = 5, PR_ck_i = 13,                 #<- Placeholder 
                 PR_pool_func = nn.MaxPool2d,
                 PR_pk_base = 2, PR_pk_s = 2, 

                 O_mods = 3,
                 O_cf_pu_m = 1, O_dropout = 0.2, 

                 cf_ns = 1, #####
                 activations = nn.ReLU(), activation_f = None,
                 batchnorm = 'before'):
        
        super(ChampionNN, self).__init__()
        
        #----------------------------------------------------------

        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]: 
            kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = dim_i
        dim_kE_width = 1 
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {'dim_i': dim_i_ref, 'dim_f': dim_kE,
                         'mods': 0, 'mods_ns': 0,    
                         'cf_i': None, 'cf_ns': kE_cf_ns, 'ck_base': kE_ck_base,
                         'doublestranded': False, 'OneByOne': kE_OneByOne,        
                         'ck_grouped': kE_ck_grouped,
                         'activations': activations, 'activation_f': None,
                         'batchnorm': batchnorm, 'dropout': None, 'bias': True,
                         'out': False}
        
        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False
        
        #----------------------------------------------------------

        mid_f = int(PR_cf_m * dim_kE[0])
        dim_mid = (mid_f, 1, 1)

        PR_ck_i = None

        PR_args = {'dim_i': dim_kE, 'dim_f': dim_mid,
                   
                   'cf_ns': cf_ns,
                   'ck_i': PR_ck_i, 'ck_base': PR_ck_base, 

                   'skip_first_conv': True,

                   'pool_func': PR_pool_func, 
                   'pk_base': PR_pk_base, 'pk_s': PR_pk_s, 
                   'activations': activations, 'batchnorm': batchnorm}
        
        self.PR = nn.Sequential(*tp.Prong_W(**PR_args))

        O_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                   'mods': O_mods,
                   'cf_ns': cf_ns, 'cf_pu': int(O_cf_pu_m * mid_f),
                   'activations': activations, 'activation_f': activation_f,
                   'batchnorm': batchnorm,
                   'dropout': O_dropout, 'out': True}
        
        self.O = nn.Sequential(*tp.Prong_X(**O_args))
        
    def forward(self, x):

        x = self.kE(x)
        x = self.PR(x)
        x = self.O(x)

        return x
    


champ_args = {'dim_i': (1,249,4), 
              'dim_f': (1,1,1),

              'kE_k': 11,
              'kE_cf_m': 100, 
              'kE_cf_ns': 1,
              'kE_ck_base': 7,
              'kE_OneByOne': False,
              
              'PR_ck_i': 13, 'PR_ck_base': 5,
              'PR_pool_func': nn.MaxPool2d,
              'PR_pk_base': 2, 'PR_pk_s': 2, 

              'O_mods': 3,'O_cf_pu_m': 1, 'O_dropout': 0.2, 

              'cf_ns': 1,
            
              'activations': nn.ReLU(), 'activation_f': None, 'batchnorm': 'before'}

champ_dict = {'dim_i': [[(1,249,4)],'cat'],
              'dim_f': [[(1,1,1)],'cat'],

              'kE_k': [[11, 13, 15], 'cat'],
              'kE_cf_m': [[100, 150, 200, 250], 'cat'],
              'kE_cf_ns': [[1, 1.15, 1.3], 'cat'],
              'kE_ck_base': [[7, 9, None], 'cat'],
              'kE_OneByOne': [[True, False], 'cat'],

            'PR_cf_m': [[1.5, 2], 'cat'],
              
              'PR_ck_i': [[8, 11, 14, 17, 20], 'cat'],
              'PR_ck_base': [[5, 7, 9], 'cat'],
              
              'PR_pool_func': [[nn.MaxPool2d, nn.AvgPool2d],'cat'],
              'PR_pk_base': [[0, 2, 4], 'cat'],
              'PR_pk_s': [[2, 3, 4], 'cat'],

              
              'O_mods': [[1, 2, 3], 'cat'],
              'O_cf_pu_m': [[0.5, 1, 2], 'cat'], 
              'O_dropout': [[0, 0.1, 0.2], 'cat'], 

              'cf_ns': [[1, 1.1, 1.2], 'cat'],              
              'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
              'activation_f': [[None], 'cat'],
              'batchnorm': [[None, 'before'], 'cat']}


def Reset_ChampionNN(champnn):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 

    lconvs, ldens = len(champnn.P1), len(champnn.P2)

    for i in np.arange(lconvs): 
        if isinstance(champnn.P1[i], nn.Conv2d): 
            champnn.P1[i].reset_parameters()
    
    for i in np.arange(ldens): 
        if isinstance(champnn.P2[i], nn.Conv2d): 
            champnn.P2[i].reset_parameters()

    return champnn



class ChampionNN_Trunc(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 

                 kE_k = 15, kE_cf_m = 100, kE_cf_ns = 1, 
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 
                 kE_freeze = False,

                 PR_cf_m = 2, PR_ck_base = 5, PR_ck_i = 13,             # <- PLACEHOLDER
                 PR_pool_func = nn.MaxPool2d,
                 PR_pk_base = 2, PR_pk_s = 2, 

                 O_mods = 3, O_cf_pu_m = 1, O_dropout = 0.2,  #<< PLACEHOLDER ARGS

                 cf_ns = 1, #####
                 activations = nn.ReLU(),
                 batchnorm = 'before'):
        
        super(ChampionNN_Trunc, self).__init__()
        
        #----------------------------------------------------------

        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]: 
            kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = dim_i
        dim_kE_width = 1 
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {'dim_i': dim_i_ref, 'dim_f': dim_kE,
                         'mods': 0, 'mods_ns': 0,    
                         'cf_i': None, 'cf_ns': kE_cf_ns, 'ck_base': kE_ck_base,
                         'doublestranded': False, 'OneByOne': kE_OneByOne,        
                         'ck_grouped': kE_ck_grouped,
                         'activations': activations, 'activation_f': None,
                         'batchnorm': batchnorm, 'dropout': None, 'bias': True,
                         'out': False}
        
        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False
        
        #----------------------------------------------------------

        mid_f = PR_cf_m * dim_kE[0]
        dim_mid = (mid_f, 1, 1)

        PR_ck_i = None

        PR_args = {'dim_i': dim_kE, 'dim_f': dim_mid,
                   
                   'cf_ns': cf_ns,
                   'ck_i': PR_ck_i, 'ck_base': PR_ck_base, 

                   'skip_first_conv': True,

                   'pool_func': PR_pool_func, 
                   'pk_base': PR_pk_base, 'pk_s': PR_pk_s, 
                   'activations': activations, 'batchnorm': batchnorm}
        
        self.PR = nn.Sequential(*tp.Prong_W(**PR_args))

        self.O = nn.Identity()
        
    def forward(self, x):

        x = self.kE(x)
        x = self.PR(x)
        x = self.O(x)

        return x



class DualChampionNN(nn.Module): 
    
    def __init__(self, 
                 dim_f = (1,1,1),

                 late_comb = True,

                 A_dim_i = (1,249,4),
                 A_kE_k = 15, A_kE_cf_m = 100, A_kE_ck_base = 4,
                 A_PR_ck_base = 5, A_PR_pk_base = 2, A_PR_pk_s = 2, 

                 B_dim_i = (1,249,4),
                 B_kE_k = 15, B_kE_cf_m = 100, B_kE_ck_base = 4,
                 B_PR_ck_base = 5, B_PR_pk_base = 2, B_PR_pk_s = 2, 

                cf_ns = 1,
                kE_freeze = False, kE_OneByOne = False,
                PR_pool_func = nn.MaxPool2d, PR_cf_m = 2, 
                
                O_mods = 3, O_cf_pu_m = 2, O_dropout = 0,

                activations = nn.ReLU(), activation_f = None, 
                batchnorm = 'before'):
        
        super(DualChampionNN, self).__init__()

        kE_ck_grouped = False

        champ_shared_args = {'dim_f': dim_f,
                             'cf_ns': cf_ns,
                             'kE_freeze': kE_freeze, 'kE_ck_grouped': kE_ck_grouped, 
                             'kE_OneByOne': kE_OneByOne, 
                             'PR_pool_func': PR_pool_func, 'PR_cf_m': PR_cf_m, 
                             'O_mods': O_mods, 'O_cf_pu_m': O_cf_pu_m,
                             'O_dropout': O_dropout,
                             'activations': activations,'batchnorm': batchnorm}
        
        champ_A_args = {'dim_i': A_dim_i, 
                        'kE_k': A_kE_k, 'kE_cf_m': A_kE_cf_m, 'kE_cf_ns': cf_ns, 
                        'kE_ck_base': A_kE_ck_base,
                        'PR_ck_base': A_PR_ck_base, 'PR_pk_base': A_PR_pk_base, 'PR_pk_s': A_PR_pk_s, 
                        **champ_shared_args}
        
        champ_B_args = {'dim_i': B_dim_i, 
                        'kE_k': B_kE_k, 'kE_cf_m': B_kE_cf_m, 'kE_cf_ns': cf_ns, 
                        'kE_ck_base': B_kE_ck_base,
                        'PR_ck_base': B_PR_ck_base, 'PR_pk_base': B_PR_pk_base, 'PR_pk_s': B_PR_pk_s, 
                        **champ_shared_args}
        
        self.late_comb = late_comb

        if late_comb:
            self.ChampA = ChampionNN(**champ_A_args)
            self.ChampB = ChampionNN(**champ_B_args)
            self.O = nn.Identity()
        
        else:
            self.ChampA = ChampionNN_Trunc(**champ_A_args)
            self.ChampB = ChampionNN_Trunc(**champ_B_args)

            A_kE_f = A_kE_cf_m * A_dim_i[0]
            A_mid_f = PR_cf_m * A_kE_f

            B_kE_f = B_kE_cf_m * B_dim_i[0]
            B_mid_f = PR_cf_m * B_kE_f

            AB_mid_f = A_mid_f + B_mid_f
            dim_mid = (AB_mid_f, 1, 1)

            O_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                   'mods': O_mods,
                   'cf_ns': cf_ns, 'cf_pu': O_cf_pu_m * AB_mid_f,
                   'activations': activations,
                   'batchnorm': batchnorm,
                   'dropout': O_dropout, 'out': True}
        
            self.O = nn.Sequential(*tp.Prong_X(**O_args))
        
        self.FA = activation_f if activation_f != None else nn.Identity()



    def forward(self, x1, x2):
        
        x1 = self.ChampA(x1)
        x2 = self.ChampB(x2)

        if self.late_comb: xcomb = x1 + x2
        else: xcomb = torch.cat([x1, x2], dim = 1)

        xcomb = self.O(xcomb)

        return self.FA(xcomb)
    


dualchamp_dict = {
    
    'A_dim_i': [[(1,500,4)],'cat'], #Set as a single dim_i
    'B_dim_i': [[(11,40,1)],'cat'], #Set as a single dim_i
    'dim_f': [[(1,1,1)],'cat'], #Set as a single dim_f

    'late_comb': [[False, True], 'cat'],



              'A_kE_k': [[11, 13, 15], 'cat'],
              'A_kE_cf_m': [[100, 150, 200], 'cat'],
              'A_kE_ck_base': [[7, 9, None], 'cat'],
              'A_PR_ck_base': [[5, 7, 9], 'cat'],
              'A_PR_pk_base': [[0, 2, 4], 'cat'],
              'A_PR_pk_s': [[2, 3, 4], 'cat'],
            
            'B_kE_k': [[5, 10, 15], 'cat'],
              'B_kE_cf_m': [[10, 20, 30], 'cat'],
              'B_kE_ck_base': [[None, 5, 7, 9], 'cat'],
              'B_PR_ck_base': [[3, 4], 'cat'],
              'B_PR_pk_base': [[0, 2], 'cat'],
              'B_PR_pk_s': [[2, 3, 4], 'cat'],


            'cf_ns': [[1, 1.1, 1.2], 'cat'],
            'kE_OneByOne': [[True, False], 'cat'],
            'PR_pool_func': [[nn.MaxPool2d, nn.AvgPool2d],'cat'],
            'PR_cf_m': [[1.5, 2], 'cat'],

              'O_mods': [[1, 2, 3], 'cat'],
              'O_cf_pu_m': [[0.5, 1, 2], 'cat'], 
              'O_dropout': [[0, 0.1, 0.2], 'cat'], 

              'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
              'activation_f': [[None], 'cat'],
              'batchnorm': [[None, 'before'], 'cat']}


def Reset_DualChampionNN(dualchamp):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 


    for champ in [dualchamp.ChampA, dualchamp.ChampB]:
    

        lke, lps, los = len(champ.kE), len(champ.PR), len(champ.O)

        for i in np.arange(lke): 
            if isinstance(champ.kE[i], nn.Conv2d): 
                champ.kE[i].reset_parameters()
        
        for i in np.arange(lps): 
            if isinstance(champ.PR[i], nn.Conv2d): 
                champ.PR[i].reset_parameters()

        if los > 1:
            for i in np.arange(los): 
                if isinstance(champ.O[i], nn.Conv2d): 
                    champ.O[i].reset_parameters()
        

    lex = len(dualchamp.O)

    if lex > 1:
        for i in np.arange(lex): 
            if isinstance(dualchamp.O[i], nn.Conv2d): 
                dualchamp.O[i].reset_parameters()

        print('done reset mod')

    return dualchamp


#=====================================

'''Cycle: Applies RNN for sequence embedding'''

class CycleNN(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 

                 kE_k = 15, kE_cf_m = 100, kE_cf_ns = 1, 
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 
                 kE_freeze = False,


                 PR_pool_func = nn.MaxPool2d, P_ck = 50,
                 PR_cf_m = 2, 
                 PR_layer = nn.GRU, PR_num_layers = 1,
                 PR_bi = True,

                 O_mods = 3,
                 O_cf_pu_m = 1, O_dropout = 0.2, 

                 cf_ns = 1, #####
                 activations = nn.ReLU(), activation_f = None,
                 batchnorm = 'before'):
        
        super(CycleNN, self).__init__()
        
        #----------------------------------------------------------

        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]: 
            kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = dim_i
        dim_kE_width = 1 
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {'dim_i': dim_i_ref, 'dim_f': dim_kE,
                         'mods': 0, 'mods_ns': 0,    
                         'cf_i': None, 'cf_ns': kE_cf_ns, 'ck_base': kE_ck_base,
                         'doublestranded': False, 'OneByOne': kE_OneByOne,        
                         'ck_grouped': kE_ck_grouped,
                         'activations': activations, 'activation_f': None,
                         'batchnorm': batchnorm, 'dropout': None, 'bias': True,
                         'out': False}
        
        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False
        
        #----------------------------------------------------------

        P_func = PR_pool_func

        if P_ck == None: P_ck = 0
        if P_ck == 0:
            # Now this means do not pool. 
            self.P, P_length = (nn.Identity(), dim_kE_length)
        elif P_ck > 1: 
            needed = P_ck - (dim_kE_length % P_ck)
            pad2add = 0
            self.P = P_func((P_ck, 1), stride = (P_ck,1), 
                            padding = (pad2add, 0), ceil_mode = True)
            P_length = int(np.ceil(dim_kE_length / P_ck))
        else: # means its 1, means we global pool
            self.P = P_func((dim_kE_length, 1), stride = (1,1), 
                            padding = (0, 0), ceil_mode = True)
            P_length = 1

        mid_f = int(PR_cf_m * dim_kE[0])
        hid_f = mid_f
        if PR_bi: 
            hid_f = mid_f // 2 
            mid_f = hid_f * 2

        dim_mid = (mid_f, 1, 1)

        self.PR_bi = PR_bi

        self.PR = PR_layer(input_size = dim_kE[0], hidden_size = int(hid_f), 
                                         num_layers = PR_num_layers, 
                                         bidirectional = PR_bi, 
                                         batch_first = True)

        #--------------------------------------------------------

        O_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                   'mods': O_mods,
                   'cf_ns': cf_ns, 'cf_pu': O_cf_pu_m * mid_f,
                   'activations': activations, 'activation_f': activation_f,
                   'batchnorm': batchnorm,
                   'dropout': O_dropout, 'out': True}
        
        self.O = nn.Sequential(*tp.Prong_X(**O_args))
        
    def forward(self, x):
        # x: (N, 1, L, 4)
        x = self.kE(x)  # (N, F, L', 1)
        x = self.P(x)

        #------------------------------

        x = x.squeeze(-1)  # (N, F, L')
        x = x.permute(0, 2, 1)  # (N, L', F)

        rnn = self.PR
        if isinstance(rnn, nn.LSTM): output, (h_n, c_n) = rnn(x)
        elif isinstance(rnn, nn.GRU): output, h_n = rnn(x)

        # h_n: (num_layers * num_directions, N, hidden_size)
        num_directions = 2 if self.PR_bi else 1

        # Get last layer's hidden state(s)
        if num_directions == 1: 
            x = h_n[-1]  # (N, hidden_size)
        else:
            # Concatenate last layer's forward and backward hidden states
            x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (N, 2*hidden_size)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (N, G, 1, 1)

        #------------------------------

        x = self.O(x)
        return x
    
def Reset_CycleNN(cyclenn):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 

    lconvs, ldens = len(cyclenn.kE), len(cyclenn.O)

    for i in np.arange(lconvs): 
        if isinstance(cyclenn.kE[i], nn.Conv2d): 
            cyclenn.kE[i].reset_parameters()
    
    for i in np.arange(ldens): 
        if isinstance(cyclenn.O[i], nn.Conv2d): 
            cyclenn.O[i].reset_parameters()
    
    cyclenn.PR.reset_parameters()

    return cyclenn


# cycle_args = {'dim_i': (1,249,4), 
#               'dim_f': (1,1,1),

#               'kE_k': 11,
#               'kE_cf_m': 100, 
#               'kE_cf_ns': 1,
#               'kE_ck_base': 7,
#               'kE_OneByOne': False,
              
#               'PR_ck_i': 13, 'PR_ck_base': 5,
#               'PR_pool_func': nn.MaxPool2d,
#               'PR_pk_base': 2, 'PR_pk_s': 2, 

#               'O_mods': 3,'O_cf_pu_m': 1, 'O_dropout': 0.2, 

#               'cf_ns': 1,
            
#               'activations': nn.ReLU(), 'activation_f': None, 'batchnorm': 'before'}

# cycle_dict = {'dim_i': [[(1,249,4)],'cat'],
#               'dim_f': [[(1,1,1)],'cat'],

#               'kE_k': [[11, 13, 15], 'cat'],
#               'kE_cf_m': [[100, 150, 200, 250], 'cat'],
#               'kE_cf_ns': [[1, 1.15, 1.3], 'cat'],
#               'kE_ck_base': [[7, 9, None], 'cat'],
#               'kE_OneByOne': [[True, False], 'cat'],

#             'PR_cf_m': [[1.5, 2], 'cat'],
              
#               'PR_ck_i': [[8, 11, 14, 17, 20], 'cat'],
#               'PR_ck_base': [[5, 7, 9], 'cat'],
              
#               'PR_pool_func': [[nn.MaxPool2d, nn.AvgPool2d],'cat'],
#               'PR_pk_base': [[0, 2, 4], 'cat'],
#               'PR_pk_s': [[2, 3, 4], 'cat'],

              
#               'O_mods': [[1, 2, 3], 'cat'],
#               'O_cf_pu_m': [[0.5, 1, 2], 'cat'], 
#               'O_dropout': [[0, 0.1, 0.2], 'cat'], 

#               'cf_ns': [[1, 1.1, 1.2], 'cat'],              
#               'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
#               'activation_f': [[None], 'cat'],
#               'batchnorm': [[None, 'before'], 'cat']}



class CycleNN_Trunc(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 

                 kE_k = 15, kE_cf_m = 100,
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 
                 kE_freeze = False,


                 PR_pool_func = nn.MaxPool2d, P_ck = 50,
                 PR_cf_m = 2, 
                 PR_layer = nn.GRU, PR_num_layers = 1,
                 PR_bi = True,

                 O_mods = 3,
                 O_cf_pu_m = 1, O_dropout = 0.2, #<< PLACEHOLDER ARGS

                 cf_ns = 1, #####
                 activations = nn.ReLU(), activation_f = None,
                 batchnorm = 'before'):
        
        super(CycleNN_Trunc, self).__init__()
        
        #----------------------------------------------------------

        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]: 
            kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = dim_i
        dim_kE_width = 1 
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {'dim_i': dim_i_ref, 'dim_f': dim_kE,
                         'mods': 0, 'mods_ns': 0,    
                         'cf_i': None, 'cf_ns': cf_ns, 'ck_base': kE_ck_base,
                         'doublestranded': False, 'OneByOne': kE_OneByOne,        
                         'ck_grouped': kE_ck_grouped,
                         'activations': activations, 'activation_f': None,
                         'batchnorm': batchnorm, 'dropout': None, 'bias': True,
                         'out': False}
        
        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False
        
        #----------------------------------------------------------

        P_func = PR_pool_func

        if P_ck == None: P_ck = 0
        if P_ck == 0:
            # Now this means do not pool. 
            self.P, P_length = (nn.Identity(), dim_kE_length)
        elif P_ck > 1: 
            needed = P_ck - (dim_kE_length % P_ck)
            pad2add = 0
            self.P = P_func((P_ck, 1), stride = (P_ck,1), 
                            padding = (pad2add, 0), ceil_mode = True)
            P_length = int(np.ceil(dim_kE_length / P_ck))
        else: # means its 1, means we global pool
            self.P = P_func((dim_kE_length, 1), stride = (1,1), 
                            padding = (0, 0), ceil_mode = True)
            P_length = 1

        mid_f = int(PR_cf_m * dim_kE[0])
        hid_f = mid_f
        if PR_bi: 
            hid_f = mid_f // 2 
            mid_f = hid_f * 2

        dim_mid = (mid_f, 1, 1)

        self.PR_bi = PR_bi

        self.PR = PR_layer(input_size = dim_kE[0], hidden_size = int(hid_f), 
                                         num_layers = PR_num_layers, 
                                         bidirectional = PR_bi, 
                                         batch_first = True)

        #--------------------------------------------------------

        self.O = nn.Identity()
        
    def forward(self, x):
        # x: (N, 1, L, 4)
        x = self.kE(x)  # (N, F, L', 1)
        x = self.P(x)

        #------------------------------

        x = x.squeeze(-1)  # (N, F, L')
        x = x.permute(0, 2, 1)  # (N, L', F)

        rnn = self.PR
        if isinstance(rnn, nn.LSTM): output, (h_n, c_n) = rnn(x)
        elif isinstance(rnn, nn.GRU): output, h_n = rnn(x)

        # h_n: (num_layers * num_directions, N, hidden_size)
        num_directions = 2 if self.PR_bi else 1

        # Get last layer's hidden state(s)
        if num_directions == 1: 
            x = h_n[-1]  # (N, hidden_size)
        else:
            # Concatenate last layer's forward and backward hidden states
            x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (N, 2*hidden_size)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (N, G, 1, 1)

        #------------------------------

        x = self.O(x)
        return x
    

class DualCycleNN(nn.Module): 
    
    def __init__(self, 
                 dim_f = (1,1,1),

                 late_comb = True,

                 A_dim_i = (1,500,4),
                 A_kE_k = 15, A_kE_cf_m = 100, A_kE_ck_base = 4,
                 A_P_ck = 50,
                 

                 B_dim_i = (11, 40, 1),
                 B_kE_k = 5, B_kE_cf_m = 10, B_kE_ck_base = 5,
                 B_P_ck = 5, 
                
                cf_ns = 1,
                kE_freeze = False, kE_OneByOne = False,
                
                PR_pool_func = nn.MaxPool2d, PR_cf_m = 2, 
                PR_layer = nn.GRU, PR_num_layers = 1,
                PR_bi = True,

                O_mods = 3, O_cf_pu_m = 2, O_dropout = 0,

                activations = nn.ReLU(), activation_f = None, 
                batchnorm = 'before'):
        
        super(DualCycleNN, self).__init__()

        kE_ck_grouped = False

        cycle_shared_args = {'dim_f': dim_f,
                             'cf_ns': cf_ns,
                             'kE_freeze': kE_freeze, 'kE_ck_grouped': kE_ck_grouped, 
                             'kE_OneByOne': kE_OneByOne, 'kE_cf_ns': cf_ns, 
                             
                             'PR_pool_func': PR_pool_func, 'PR_cf_m': PR_cf_m, 
                             'PR_layer': PR_layer, 'PR_num_layers': PR_num_layers, 
                             'PR_bi': PR_bi,

                             'O_mods': O_mods, 'O_cf_pu_m': O_cf_pu_m, 'O_dropout': O_dropout,
                             'activations': activations,'batchnorm': batchnorm}
        
        cycle_A_args = {'dim_i': A_dim_i, 
                        'kE_k': A_kE_k, 'kE_cf_m': A_kE_cf_m,
                        'kE_ck_base': A_kE_ck_base,
                        'P_ck': A_P_ck,                         
                        **cycle_shared_args}
        
        cycle_B_args = {'dim_i': B_dim_i, 
                        'kE_k': B_kE_k, 'kE_cf_m': B_kE_cf_m,
                        'kE_ck_base': B_kE_ck_base,
                        'P_ck': B_P_ck,        
                        **cycle_shared_args}
        
        self.late_comb = late_comb

        if late_comb:
            self.CycleA = CycleNN(**cycle_A_args)
            self.CycleB = CycleNN(**cycle_B_args)
            self.O = nn.Identity()
        
        else:
            self.CycleA = CycleNN_Trunc(**cycle_A_args)
            self.CycleB = CycleNN_Trunc(**cycle_B_args)

            A_kE_f = int(A_kE_cf_m * A_dim_i[0])
            A_mid_f = int(PR_cf_m * A_kE_f)

            B_kE_f = int(B_kE_cf_m * B_dim_i[0])
            B_mid_f = int(PR_cf_m * B_kE_f)

            AB_mid_f = A_mid_f + B_mid_f
            dim_mid = (AB_mid_f, 1, 1)

            O_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                   'mods': O_mods,
                   'cf_ns': cf_ns, 'cf_pu': int(O_cf_pu_m * AB_mid_f),
                   'activations': activations,
                   'batchnorm': batchnorm,
                   'dropout': O_dropout, 'out': True}
        
            self.O = nn.Sequential(*tp.Prong_X(**O_args))
        
        self.FA = activation_f if activation_f != None else nn.Identity()



    def forward(self, x1, x2):
        
        x1 = self.CycleA(x1)
        x2 = self.CycleB(x2)

        if self.late_comb: xcomb = x1 + x2
        else: xcomb = torch.cat([x1, x2], dim = 1)

        xcomb = self.O(xcomb)

        return self.FA(xcomb)
    



#=====================================

'''Ultra (Mimic): Combines all types of sequence embedding into one architecture '''


class Ultra(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),

                 trunc = False,
                 
                 kE_k = 15, kE_cf_m = 100,
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 
                 kE_freeze = False,

                 #--------------------------------------------

                 PR_cf_m = 2, PR_ck_base = 5,
                 PR_pool_func = nn.AvgPool2d,
                 PR_pk_base = 2, PR_pk_s = 2, 

                 P_ck = 50, PR_layer = None, 
                 PR_num_layers = 1, PR_bi = True, # <- RNN arguments

                 #--------------------------------------------

                 O_mods = 3, O_cf_pu_m = 1, O_dropout = 0.2, 

                 cf_ns = 1, #####
                 activations = nn.ReLU(), activation_f = None,
                 batchnorm = 'before'):
        
        super(Ultra, self).__init__()

        self.trunc = trunc
        
        #----------------------------------------------------------

        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]: 
            kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = dim_i
        dim_kE_width = 1 
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {'dim_i': dim_i_ref, 'dim_f': dim_kE,
                         'mods': 0, 'mods_ns': 0,    
                         'cf_i': None, 'cf_ns': cf_ns, 'ck_base': kE_ck_base,
                         'doublestranded': False, 'OneByOne': kE_OneByOne,        
                         'ck_grouped': kE_ck_grouped,
                         'activations': activations, 'activation_f': None,
                         'batchnorm': batchnorm, 'dropout': None, 'bias': True,
                         'out': False}
        
        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False
        
        #----------------------------------------------------------

        P_func = PR_pool_func

        if P_ck == None: P_ck = 0
        if P_ck == 0:
            # Now this means do not pool. 
            self.P, P_length = (nn.Identity(), dim_kE_length)
        elif P_ck > 1: 
            needed = P_ck - (dim_kE_length % P_ck)
            pad2add = 0
            self.P = P_func((P_ck, 1), stride = (P_ck,1), 
                            padding = (pad2add, 0), ceil_mode = True)
            P_length = int(np.ceil(dim_kE_length / P_ck))
        else: # means its 1, means we global pool
            self.P = P_func((dim_kE_length, 1), stride = (1,1), 
                            padding = (0, 0), ceil_mode = True)
            P_length = 1
        
        #----------------------------------------------------------

        self.rnn = False

        mid_f = int(PR_cf_m * dim_kE[0])
        dim_mid = (mid_f, 1, 1)

        if O_mods == 0: 
            self.PR = nn.Identity()
            dim_mid = (dim_kE[0], P_length, 1)
            mid_f = dim_kE[0]

        elif PR_layer is None: 
            
            skip_first_conv = True
            if P_length != dim_kE[1]: skip_first_conv = False

            dim_kE = (dim_kE[0], P_length, 1)

            # Use CPMs 
            PR_args = {'dim_i': dim_kE, 'dim_f': dim_mid, 'cf_ns': cf_ns, 
                       'ck_i': None, 'skip_first_conv': skip_first_conv,
                       'ck_base': PR_ck_base, 'pk_base': PR_pk_base, 'pk_s': PR_pk_s, 
                       'activations': activations, 'batchnorm': batchnorm}
            
            self.PR = nn.Sequential(*tp.Prong_W(**PR_args))


        elif PR_layer in [nn.GRU, nn.LSTM]:
            
            self.rnn = True
            hid_f = mid_f
            if PR_bi: 
                hid_f = mid_f // 2 
                mid_f = hid_f * 2
            dim_mid = (mid_f, 1, 1)
            self.PR_bi = PR_bi
            self.PR = PR_layer(input_size = dim_kE[0], hidden_size = int(hid_f), 
                                            num_layers = PR_num_layers, 
                                            bidirectional = PR_bi, 
                                            batch_first = True)
        elif PR_layer in [nn.Conv2d]: 
            self.PR = nn.Conv2d(dim_kE[0], mid_f, kernel_size = (P_length, 1), bias = True, 
                                stride = (1, 1), padding = 0)
        
        else: 
            dim_mid = (dim_kE[0], 1, 1)                 # <- Not using PR_cf_m here. 
            self.PR = PR_layer(embed_dim = dim_kE[0])
            mid_f = dim_kE[0]



        #--------------------------------------------------------

        if trunc: 
            self.O = nn.Identity()
        else: 

            O_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                    'mods': O_mods,
                    'cf_ns': cf_ns, 'cf_pu': O_cf_pu_m * mid_f,
                    'activations': activations, 'activation_f': activation_f,
                    'batchnorm': batchnorm,
                    'dropout': O_dropout, 'out': True}
            
            self.O = nn.Sequential(*tp.Prong_X(**O_args))
        
    def forward(self, x):
        # x: (N, 1, L, 4)
        x = self.kE(x)  # (N, F, L', 1)
        x = self.P(x)

        #------------------------------
        if self.rnn: 
            x = x.squeeze(-1)  # (N, F, L')
            x = x.permute(0, 2, 1)  # (N, L', F)

            rnn = self.PR
            if isinstance(rnn, nn.LSTM): _, (h_n, _) = rnn(x)
            elif isinstance(rnn, nn.GRU): _, h_n = rnn(x)
            # h_n: (num_layers * num_directions, N, hidden_size)
            num_directions = 2 if self.PR_bi else 1
            # Get last layer's hidden state(s)
            if num_directions == 1: 
                x = h_n[-1]  # (N, hidden_size)
            else:
                # Concatenate last layer's forward and backward hidden states
                x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (N, 2*hidden_size)
            x = x.unsqueeze(-1).unsqueeze(-1)  # (N, G, 1, 1)
        
        else: 
            x = self.PR(x)
        #------------------------------

        return self.O(x)

        
    
def Reset_Ultra(ultra):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 

    lconvs, lprs, ldens = len(ultra.kE), len(ultra.PR), len(ultra.O)

    for i in np.arange(lconvs): 
        if isinstance(ultra.kE[i], nn.Conv2d): 
            ultra.kE[i].reset_parameters() 
    
    if ldens > 1:  
        for i in np.arange(ldens): 
            if isinstance(ultra.O[i], nn.Conv2d): 
                ultra.O[i].reset_parameters()
    
    if lprs > 1:
        for i in np.arange(lprs): 
            if isinstance(ultra.PR[i], nn.Conv2d): 
                ultra.PR[i].reset_parameters()
    elif lprs == 1: 
        ultra.PR.reset_parameters()

    return ultra







class Dual_Ultra(nn.Module): 
    
    def __init__(self, 
                 dim_f = (1,1,1),

                 late_comb = True,

                 A_dim_i = (1,249,4),
                 A_kE_k = 15, A_kE_cf_m = 100, A_kE_ck_base = 4,
                 A_PR_ck_base = 5, A_PR_pk_base = 2, A_PR_pk_s = 2, 
                 A_P_ck = None,

                 B_dim_i = (1,249,4),
                 B_kE_k = 15, B_kE_cf_m = 100, B_kE_ck_base = 4,
                 B_PR_ck_base = 5, B_PR_pk_base = 2, B_PR_pk_s = 2, 
                 B_P_ck = None,

                 #--------------------------------------------------
                 PR_layer = None, PR_num_layers = 1, PR_bi = True,
                
                cf_ns = 1,
                kE_freeze = False, kE_OneByOne = False,
                PR_pool_func = nn.MaxPool2d, PR_cf_m = 2, 
                
                O_mods = 3, O_cf_pu_m = 2, O_dropout = 0,

                activations = nn.ReLU(), activation_f = None, 
                batchnorm = 'before'):
        
        super(Dual_Ultra, self).__init__()

        trunc = False if late_comb else True

        kE_ck_grouped = False

        ultra_shared_args = {'dim_f': dim_f, 'trunc': trunc,
                             'PR_layer': PR_layer, 'PR_num_layers': PR_num_layers, 'PR_bi': PR_bi,
                             'cf_ns': cf_ns,
                             'kE_freeze': kE_freeze, 'kE_ck_grouped': kE_ck_grouped, 
                             'kE_OneByOne': kE_OneByOne, 
                             'PR_pool_func': PR_pool_func, 'PR_cf_m': PR_cf_m, 
                             'O_mods': O_mods, 'O_cf_pu_m': O_cf_pu_m,
                             'O_dropout': O_dropout,
                             'activations': activations,'batchnorm': batchnorm}
        
        ultra_A_args = {'dim_i': A_dim_i, 
                        'kE_k': A_kE_k, 'kE_cf_m': A_kE_cf_m, 'kE_ck_base': A_kE_ck_base,
                        'P_ck': A_P_ck, 
                        'PR_ck_base': A_PR_ck_base, 'PR_pk_base': A_PR_pk_base, 'PR_pk_s': A_PR_pk_s, 
                        **ultra_shared_args}
        
        ultra_B_args = {'dim_i': B_dim_i, 
                        'kE_k': B_kE_k, 'kE_cf_m': B_kE_cf_m,'kE_ck_base': B_kE_ck_base,
                        'P_ck': B_P_ck, 
                        'PR_ck_base': B_PR_ck_base, 'PR_pk_base': B_PR_pk_base, 'PR_pk_s': B_PR_pk_s, 
                        **ultra_shared_args}
        
        self.late_comb = late_comb

        if late_comb:
            self.ultraA = Ultra(**ultra_A_args)
            self.ultraB = Ultra(**ultra_B_args)
            self.O = nn.Identity()
        
        else:
            self.ultraA = Ultra(**ultra_A_args)
            self.ultraB = Ultra(**ultra_B_args)

            if PR_layer is not None: 
                if PR_layer not in [nn.GRU, nn.LSTM, nn.Conv2d]: 
                    PR_cf_m = 1               

            A_kE_f = A_kE_cf_m * A_dim_i[0]
            A_mid_f = PR_cf_m * A_kE_f

            B_kE_f = B_kE_cf_m * B_dim_i[0]
            B_mid_f = PR_cf_m * B_kE_f

            if PR_layer in [nn.GRU, nn.LSTM] and PR_bi: 
                hid_f = A_mid_f // 2 
                A_mid_f = hid_f * 2
                
                hid_f = B_mid_f // 2 
                B_mid_f = hid_f * 2

            AB_mid_f = int(A_mid_f + B_mid_f)
            dim_mid = (AB_mid_f, 1, 1)

            O_args = {'dim_i': dim_mid, 'dim_f': dim_f,
                   'mods': O_mods,
                   'cf_ns': cf_ns, 'cf_pu': int(O_cf_pu_m * AB_mid_f),
                   'activations': activations,
                   'batchnorm': batchnorm,
                   'dropout': O_dropout, 'out': True}
        
            self.O = nn.Sequential(*tp.Prong_X(**O_args))
        
        self.FA = activation_f if activation_f != None else nn.Identity()



    def forward(self, x1, x2):
        
        x1 = self.ultraA(x1)
        x2 = self.ultraB(x2)

        if self.late_comb: xcomb = x1 + x2
        else: xcomb = torch.cat([x1, x2], dim = 1)

        xcomb = self.O(xcomb)

        return self.FA(xcomb)
    



def Reset_Dual_Ultra(dual_ultra):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 


    for ultra in [dual_ultra.ultraA, dual_ultra.ultraB]:
        Reset_Ultra(ultra)
    
    lex = len(dual_ultra.O)

    if lex > 1:
        for i in np.arange(lex): 
            if isinstance(dual_ultra.O[i], nn.Conv2d): 
                dual_ultra.O[i].reset_parameters()

        print('done reset mod')
    
    return dual_ultra