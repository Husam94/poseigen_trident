import numpy as np

import torch
import torch.nn as nn

import poseigen_seaside.basics as se
import poseigen_trident.utils as tu


######################################################################################################

# Prong X

def Prong_X(dim_i = (300, 200, 1), dim_f = (20, 1, 1),

            mods = 0, mods_ns = 0,            
            cf_i = None, cf_ns = 1, cf_pu = None,
            ck_base = None, 

            doublestranded = False, OneByOne = False,            
            ck_grouped = False,

            activations = nn.ReLU(), activation_f = None,
            batchnorm = 'before', dropout = None, bias = True,
            out = False): 
    
    #25.01.22 Took off reflect because its bs. 

    #d oublestranded if its receiving an input that is double stranded. It would process seperately. 
    # OneByOne forces the last conv to have a kernel length of 1. This is beneficial for kmer embedding. 
    
    if ck_base is None: ck_base = 0
    if mods is None: mods = 0
    
    if cf_i == 0: cf_i = None
    
    ##########################################################

    #To determine the kernel sizes, either we:
    # 1) ck_base which uses the same kernel to get to where you want; determines the number of mods for you. 
    # 2) mods_ns which determines the kernel size to get the length you want, must specify the number of mods
    # for 1), we can easily add a OneByOne option. 
    # for 2), to add a OneByOne opton, wehave to remove one from the number of mods used, then we add at the end. 

    if dim_i[1] == dim_f[1]: ck_base, mods_ns = 0, 0 # only use mods in this case. 

    if ck_base != 0: # Then we reduce by this amount until we get to final
        cks = []
        curlen = dim_i[1]
        while curlen > dim_f[1]: 
            if (curlen - ck_base + 1) >= dim_f[1]: 
                cks.append(ck_base)
                curlen = curlen - ck_base + 1
            else: 
                cks.append(curlen - dim_f[1] + 1)
                curlen = dim_f[1]
        if OneByOne: cks.append(1) ########
    
    elif mods_ns > 0: 
        mx = mods if OneByOne else mods + 1
        v = np.round(se.GeomNumSpacing(dim_i[1], dim_f[1], mx, mods_ns)).astype(int)
        cks = tu.FindKernelSize(v)
        if OneByOne: cks = cks + [1]
    
    else: 
        cks = [dim_i[1] - dim_f[1] + 1] + ([1] * (mods - 1))

    if mods == 0: mods = len(cks)

    ######################################################

    if mods == 0: cf_i, cf_pu = None, None

    if cf_pu is None: cf_pu = 0
    mx, dfx = (mods-1, cf_pu) if (cf_pu > 0 and mods > 2) else (mods, dim_f[0])

    cfs = np.round(se.GeomNumSpacing(dim_i[0], dfx, mx + 1, cf_ns)).astype(int)
    
    if cf_i is not None: 
        cfs = np.hstack([dim_i[0], 
                    np.round(se.GeomNumSpacing(cf_i, dfx, mx, cf_ns)).astype(int)])
    
    if cf_pu > 0: cfs = np.array(cfs.tolist() + [dim_f[0]])

    if ck_grouped > 1: num_groups = ck_grouped
    elif ck_grouped == True: num_groups = dim_i[0]
    else: num_groups = 1
    
    cfs = (cfs // num_groups) * num_groups #DIVIDING EVERYTHING MY THE INITIAL.

    #######################################

    if dropout is None: dropout = 0


    if dim_i[-1] == 1: doublestranded = False
    
    layers = []
    for i in range(mods): 

        dd, ss = (1, 1)
        if i == 0: 
            if doublestranded: dd, ss = (dim_i[-1] // 2, dim_i[-1] // 2)
            else: dd, ss = (dim_i[-1], 1)


        layers.append(nn.Conv2d(cfs[i], cfs[i+1], (cks[i], dd), 
                                groups = num_groups, 
                                padding='valid', bias = bias, stride = (1,ss)))

        lays_ex = []

        if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        if activations is not None: lays_ex.append(activations)
        if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        if dropout > 0:  lays_ex.append(nn.Dropout2d(dropout))

        if out and i == mods - 1: pass
        else: layers.extend(lays_ex)
    
    if activation_f is not None: layers.append(activation_f)

    return layers


Prong_X_args = {'dim_i': (300, 200, 1), 'dim_f': (20, 1, 1),
                'mods': 0, 'mods_ns': 0,    
                
                'cf_i': None, 'cf_ns': 1, 'cf_pu': None,
                'ck_base': 5, 
                
                'doublestranded': False, 'OneByOne': False,            
                'ck_grouped': False,
                
                'activations': nn.ReLU(), 'activation_f': None,
                'batchnorm': 'before', 'dropout': None, 'bias': True,
                'out': False}

######################################################################################################

# Prong Y

# def Prong_Y_calc(dim_i = 200, dim_o = 1,
#                  mods = 3, mods_ns = 0.5, 
#                  ck_base = 10, ck_i = None, ck_dynamic = False, pool_s2k = 0.1, 
#                  out = True, skip_first_ck = False): 
    
#     # If ck_dynamic AND ck_base == < 1 (its a proportion), the the ck == determined using the difference in lentths
#     # Can also skip the first Conv. (skip_first_ck)
#     # Can also skip the last pool [use only Conv] for OUT
    
#     #the goal here == to determine the conv k length (ck), pool k length (pk), and pool stride (ps) for each mod
#     # if ns_lenght == none, it uses a single conv filter for the entire lenght right away, no pool. 

#     #adding a ck_i for the first conv filter length for k-mer embedding. If its none or 0 then don't apply. 

#     if mods == 0 or mods == None: mods = 1

#     if ck_base == None: ck_base = 0
#     if mods_ns == None: mods_ns = 0
#     if mods_ns == 0 or ck_base == 0 :  #######################
#         nls = [dim_i] + [1]*mods
#         ck_base = dim_i
#     else: nls = np.round(se.GeomNumSpacing(dim_i,dim_o, mods + 1, mods_ns)).astype(int)

#     if ck_i == None: ck_i = 0
        
#     cks, pks, pss, ppads = [], [], [], []

#     dim_i1 = dim_i

#     for x in np.arange(mods):
        
#         dim_i2 = nls[x+1]
#         dim_diff = dim_i1 - dim_i2

#         ck = ck_base
#         if x == 0: 
#             if ck_i > 0: ck = ck_i
#             if skip_first_ck: ck = 1

        

#         if (ck < 1) and ck_dynamic: 
#             ck = np.clip(np.floor(dim_diff * ck), a_min = 1, a_max = None).astype(int)

#         if (dim_i1 - ck + 1 <= dim_i2) or ((x == mods - 1) and out): #Only Conv here.
#             ck = dim_i1 - dim_i2 + 1
#             pk, ps, ppad = 1, 1, 0
#             dim_i1 = dim_i2
        
#         else: 
#             dim_ac = dim_i1 - ck + 1
            
#             pk = np.floor(dim_ac / ((pool_s2k*(dim_i2 - 1) + 1))).astype(int)

#             ps = np.floor(pk * pool_s2k).astype(int) ##################
#             if ps < 1: ps = 1

#             ppad = (((dim_i2 - 1) * ps) - dim_ac + pk) // 2
#             if ppad < 0: ppad = 0

#             dim_i1 = np.ceil(((dim_ac - pk + 2*ppad) / ps) + 1).astype(int)
                                
#         cks.append(ck)
#         pks.append(pk)
#         pss.append(ps)
#         ppads.append(ppad)

#     return cks, pks, pss, ppads


def Prong_Y_calc(dim_i = 200, dim_o = 1,
                 mods = 3, mods_ns = 0.5, 
                 ck_base = 10, ck_i = None, ck_dynamic = False, pool_k2s = 1, 
                 out = True, skip_first_ck = False): 
    
    # If ck_dynamic AND ck_base == < 1 (its a proportion), the the ck == determined using the difference in lentths
    # Can also skip the first Conv. (skip_first_ck)
    # Can also skip the last pool [use only Conv] for OUT
    
    #the goal here == to determine the conv k length (ck), pool k length (pk), and pool stride (ps) for each mod
    # if ns_lenght == none, it uses a single conv filter for the entire lenght right away, no pool. 

    #adding a ck_i for the first conv filter length for k-mer embedding. If its none or 0 then don't apply. 

    if mods == 0 or mods == None: mods = 1

    if ck_base == None: ck_base = 0
    if mods_ns == None: mods_ns = 0
    if mods_ns == 0 or ck_base == 0 :  #######################
        nls = [dim_i] + [1]*mods
        ck_base = dim_i
    else: nls = np.round(se.GeomNumSpacing(dim_i,dim_o, mods + 1, mods_ns)).astype(int)

    if ck_i == None: ck_i = 0
        
    cks, pks, pss = [], [], []

    dim_i1 = dim_i

    for x in np.arange(mods):
        
        dim_i2 = nls[x+1]
        dim_diff = dim_i1 - dim_i2

        ck = ck_base
        if x == 0: 
            if ck_i > 0: ck = ck_i
            if skip_first_ck: ck = 1

        if (ck < 1) and ck_dynamic: 
            ck = np.clip(np.floor(dim_diff * ck), a_min = 1, a_max = None).astype(int)

        if (dim_i1 - ck + 1 <= dim_i2) or ((x == mods - 1) and out): #Only Conv here.
            ck = dim_i1 - dim_i2 + 1
            pk, ps = 1, 1
            dim_i1 = dim_i2
        
        else: 
            dim_ac = dim_i1 - ck + 1
            ps = np.floor(dim_ac / (dim_i2 - 1 + pool_k2s)).astype(int) ##################
            pk = pk = dim_ac - ((dim_i2 - 1) * ps)

            if pk == 1: ps = 1

            dim_i1 = np.ceil(((dim_ac - pk) / ps) + 1).astype(int)
                
        cks.append(ck)
        pks.append(pk)
        pss.append(ps)


    return cks, pks, pss


def Prong_Y(dim_i = (300, 200, 1), dim_f = (20, 1, 1),
            
            mods = 3, mods_ns = 0.3,
            cf_i = None, cf_pu = None,
            cf_ns = 1, ck_base = 10, ck_i = None,

            doublestranded = False, 
            ck_dynamic = False, skip_first_ck = False,
            pool_k2s = 1, pool_func = nn.MaxPool2d, actb4pool = True,
            
            activations = nn.ReLU(), activation_f = None,
            batchnorm = 'before', dropout = None, bias = True,
            out = False): 


    #CHANGED TO HAVE DROP OUT BEFORE THE CONV!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #if mods == 0, it will go into just pooling (skips conv) 
    #if pooling kernel == 1, skip pooling 

    #cf_m (conv filt multiplier), removed. Should be taken care of elsewhere. 
    # But we could have a cf_i. 

    #cf_f_m = the penultimate layer filter, useful for final output

    if mods == 0 or mods == None: mods = 1
    if mods < 2: cf_i, cf_pu = None, None #Does not apply it. 

    #####################################################

    if cf_pu == None: cf_pu = 0
    mx, dfx = (mods-1, cf_pu) if (cf_pu > 0 and mods > 2) else (mods, dim_f[0])

    cfs = np.round(se.GeomNumSpacing(dim_i[0], dfx, mx + 1, cf_ns)).astype(int)
    
    if cf_i != None: 
        cfs = np.hstack([dim_i[0], 
                    np.round(se.GeomNumSpacing(cf_i, dfx, mx, cf_ns)).astype(int)])
    
    if cf_pu > 0: cfs = cfs.tolist() + [dim_f[0]]
        
    ################################
    
    if skip_first_ck and mods > 1: cfs[1] = cfs[0] 

    cks, pks, pss = Prong_Y_calc(dim_i = dim_i[1], dim_o = dim_f[1],
                                 mods = mods, mods_ns = mods_ns, 
                                 ck_base = ck_base, ck_i = ck_i, ck_dynamic = ck_dynamic, 
                                 pool_k2s = pool_k2s, 
                                 out = out, skip_first_ck = skip_first_ck)


    if dropout == None: dropout = 0

    if dim_i[-1] == 1: doublestranded = False
    
    layers = []
 
    for i in range(mods):

        dd, ss = (1, 1)
        if i == 0: 
            if doublestranded: dd, ss = (dim_i[-1] // 2, dim_i[-1] // 2)
            else: dd, ss = (dim_i[-1], 1)
        
        if ((i == 0) and skip_first_ck): pass
        else: 

            if dropout > 0: layers.append(nn.Dropout2d(dropout))                            #************************

            conv_layer = nn.Conv2d(cfs[i], cfs[i+1], kernel_size = (cks[i], dd), bias = bias,
                            stride = (1, ss), padding = 0)
            layers.append(conv_layer)
        
        lays_ex = []

        if actb4pool: 
            if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
            if activations != None: lays_ex.append(activations)
            if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
            
        if pks[i] > 1:    
            pool_layer = pool_func(kernel_size = (pks[i], 1), stride = (pss[i], 1),
                            padding = 0, ceil_mode = True)
            lays_ex.append(pool_layer)
        
        if actb4pool == False: 
            if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
            if activations != None: lays_ex.append(activations)
            if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        
        if ((i == 0) and skip_first_ck): lays_ex = [pool_layer]

        if out and i == mods - 1: pass
        else: layers.extend(lays_ex)
    
    if activation_f != None: layers.append(activation_f)

    return layers

Prong_Y_arg = {'dim_i': (300, 200, 1), 'dim_f': (20, 1, 1),
                
                'mods': 1, 'mods_ns': 0.3,    
                'cf_i': None, 'cf_ns': 1, 
                'ck_base': 5, 'ck_i': None, 
                
                'doublestranded': False, 
                'ck_dynamic': False, 'skip_first_ck': False,
                'pool_k2s': 1, 'pool_func': nn.MaxPool2d, 'actb4pool': True,
                
                'activations': nn.ReLU(), 'activation_f': None,
                'batchnorm': 'before', 'dropout': None, 'bias': True,
                'out': False}


######################################################################################################

# Prong Z

def Prong_Z_calc(dim_i = 10, dim_o = 200, tcf_s2k = 0.5):
    #Transpose Convolution Output Size = (Input Size - 1) * Strides + Filter Size - 2 * Padding + Ouput Padding
    X = dim_i 
    Y = dim_o
    r = tcf_s2k

    k = Y / (((X-1)*r) + 1)
    s = int(np.floor(r * k))
    if s < 1: s = 1
    k = Y - ((X-1)*s)

    if ((X-1)*s) + k > Y: s = 1

    return k, s


def Prong_Z(dim_i = (20, 1, 1), dim_f = (1, 200, 1),

            mods = 0, mods_ns = 0.3,            
            tcf_ns = 1, tcf_s2k = 0.5,

            smooth_k = None, 

            activations = nn.ReLU(), activation_f = None,
            batchnorm = 'before', dropout = None, bias = True,
            out = False): 

    if smooth_k == None or smooth_k == 0: smooth_k = 1
    dfl = dim_f[1] + smooth_k - 1

    cfs = np.round(se.GeomNumSpacing(dim_i[0], dim_f[0], mods + 1, tcf_ns)).astype(int)
    nls = np.round(se.GeomNumSpacing(dim_i[1], dfl, mods + 1, mods_ns)).astype(int)

    if dropout == None: dropout = 0
    
    layers = []
    for i in range(mods): 

        tk, ts = Prong_Z_calc(dim_i = nls[i], dim_o = nls[i+1], tcf_s2k = tcf_s2k)
        
        tconv_layer = nn.ConvTranspose2d(cfs[i], cfs[i+1], kernel_size = (tk, 1), 
                                            stride = (ts, 1), bias = bias)
        layers.append(tconv_layer)

        lays_ex = []

        if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        if activations != None: lays_ex.append(activations)
        if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        if dropout > 0: lays_ex.append(nn.Dropout2d(dropout))

        if out and i == mods - 1: pass
        else: layers.extend(lays_ex)
    

    if smooth_k > 1: 
        layers.append(nn.AvgPool2d(kernel_size = (smooth_k, 1), stride = (1,1)))

    return layers








#######################################

# def Prong_Y(dim_i = (300, 200, 1), dim_f = (20, 1, 1),
            
#             mods = 3, mods_ns = 0.3,
#             cf_i = None, cf_pu = None,
#             cf_ns = 1, ck_base = 10, ck_i = None,

#             doublestranded = False, 
#             ck_dynamic = False, skip_first_ck = False,
#             pool_k2s = 1, pool_func = nn.MaxPool2d, actb4pool = True,
            
#             activations = nn.ReLU(), activation_f = None,
#             batchnorm = 'before', dropout = None, bias = True,
#             out = False): 

#     #if mods == 0, it will go into just pooling (skips conv) 
#     #if pooling kernel == 1, skip pooling 

#     #cf_m (conv filt multiplier), removed. Should be taken care of elsewhere. 
#     # But we could have a cf_i. 

#     #cf_f_m = the penultimate layer filter, useful for final output

#     if mods == 0 or mods == None: mods = 1
#     if mods < 2: cf_i, cf_pu = None, None #Does not apply it. 

#     #####################################################

#     if cf_pu == None: cf_pu = 0
#     mx, dfx = (mods-1, cf_pu) if (cf_pu > 0 and mods > 2) else (mods, dim_f[0])

#     cfs = np.round(se.GeomNumSpacing(dim_i[0], dfx, mx + 1, cf_ns)).astype(int)
    
#     if cf_i != None: 
#         cfs = np.hstack([dim_i[0], 
#                     np.round(se.GeomNumSpacing(cf_i, dfx, mx, cf_ns)).astype(int)])
    
#     if cf_pu > 0: cfs = cfs.tolist() + [dim_f[0]]
        
#     ################################
    
#     if skip_first_ck and mods > 1: cfs[1] = cfs[0] 

#     cks, pks, pss = Prong_Y_calc(dim_i = dim_i[1], dim_o = dim_f[1],
#                                  mods = mods, mods_ns = mods_ns, 
#                                  ck_base = ck_base, ck_i = ck_i, ck_dynamic = ck_dynamic, 
#                                  pool_k2s = pool_k2s, 
#                                  out = out, skip_first_ck = skip_first_ck)


#     if dropout == None: dropout = 0

#     if dim_i[-1] == 1: doublestranded = False
    
#     layers = []
 
#     for i in range(mods):

#         dd, ss = (1, 1)
#         if i == 0: 
#             if doublestranded: dd, ss = (dim_i[-1] // 2, dim_i[-1] // 2)
#             else: dd, ss = (dim_i[-1], 1)
        
#         if ((i == 0) and skip_first_ck): pass
#         else: 
#             conv_layer = nn.Conv2d(cfs[i], cfs[i+1], kernel_size = (cks[i], dd), bias = bias,
#                             stride = (1, ss), padding = 0)
#             layers.append(conv_layer)
        
#         lays_ex = []

#         if actb4pool: 
#             if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
#             if activations != None: lays_ex.append(activations)
#             if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
            
#         if pks[i] > 1:    
#             pool_layer = pool_func(kernel_size = (pks[i], 1), stride = (pss[i], 1),
#                             padding = 0, ceil_mode = True)
#             lays_ex.append(pool_layer)
        
#         if actb4pool == False: 
#             if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
#             if activations != None: lays_ex.append(activations)
#             if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        
#         if dropout > 0: 
#             lays_ex.append(nn.Dropout2d(dropout))
        
#         if ((i == 0) and skip_first_ck): lays_ex = [pool_layer]

#         if out and i == mods - 1: pass
#         else: layers.extend(lays_ex)
    
#     if activation_f != None: layers.append(activation_f)

#     return layers

