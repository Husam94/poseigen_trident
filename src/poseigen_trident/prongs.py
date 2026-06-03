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

    if cf_pu is not None: cf_pu = int(cf_pu)

    if mods == 1: 
        cfs = [dim_i[0], dim_f[0]]
        
    elif mods == 2:
        if cf_i is not None: cfs = [dim_i[0], cf_i, dim_f[0]]
        elif cf_pu is not None: cfs = [dim_i[0], cf_pu, dim_f[0]]
        else: cfs = np.linspace(dim_i[0], dim_f[0], 3).astype(int)
    else:
        cfs = [dim_i[0]]

        # Determine positions for cf_i and cf_pu
        if cf_i is not None and cf_pu is not None and mods > 3:
            # Both cf_i and cf_pu set, interpolate between them
            cfs.append(cf_i)
            n_middle = mods - 3
            if n_middle > 0:
                cfs += list(np.linspace(cf_i, cf_pu, n_middle + 2, dtype=int)[1:-1])
            cfs.append(cf_pu)
        elif cf_i is not None:
            # Only cf_i set, interpolate from cf_i to output
            n_middle = mods - 2
            cfs.append(cf_i)
            if n_middle > 0:
                cfs += list(np.linspace(cf_i, dim_f[0], n_middle + 2, dtype=int)[1:-1])
        elif cf_pu is not None:
            # Only cf_pu set, interpolate from input to cf_pu
            n_middle = mods - 2
            if n_middle > 0:
                cfs += list(np.linspace(dim_i[0], cf_pu, n_middle + 2, dtype=int)[1:-1])
            cfs.append(cf_pu)
        else:
            # Neither set, interpolate from input to output
            cfs += list(np.linspace(dim_i[0], dim_f[0], mods + 1, dtype=int)[1:-1])
        cfs.append(dim_f[0])
    
    cfs = np.array(cfs)

    # else: 
    #     if cf_i is not None: 
    #         cfs = np.hstack([dim_i[0], 
    #                 np.round(se.GeomNumSpacing(cf_i, dfx, mx, cf_ns)).astype(int)])









    # if mods == 0: cf_i, cf_pu = None, None

    # if cf_pu is None: cf_pu = 0
    # mx, dfx = (mods-1, cf_pu) if (cf_pu > 0 and mods > 2) else (mods, dim_f[0])

    # cfs = np.round(se.GeomNumSpacing(dim_i[0], dfx, mx + 1, cf_ns)).astype(int)
    
    # if cf_i is not None: 
    #     cfs = np.hstack([dim_i[0], 
    #                 np.round(se.GeomNumSpacing(cf_i, dfx, mx, cf_ns)).astype(int)])
    
    # if cf_pu > 0: 
    #     cfs = np.round(se.GeomNumSpacing(dim_i[0], dim_i[0] * cf_pu, mx, cf_ns)).astype(int)

    #     cfs = np.hstack([cfs, dfx])






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






#------------------------------------------------


import math

def Prong_W_params(seq_len, ck_i, ck_base, pk_base, pk_s, prints=False, skip_first_conv=False):
    pk_base = pk_base + pk_s
    if ck_i is None: ck_i = ck_base

    ck_bases, pk_bases, pk_ss, pool_pads = [], [], [], []
    current_len = seq_len
    module_idx = 1

    if prints: print(f"=== Starting Sequence Length: {seq_len} ===")

    while current_len > 1:
        input_len = current_len  # Input length at start of module

        if prints: print(f"\nModule {module_idx}:")
        if prints: print(f"  Input Length: {input_len}")

        # 1. Determine conv kernel
        if skip_first_conv and module_idx == 1:
            this_conv_k = None
            conv_out_len = input_len
        else:
            target_conv_k = ck_i if module_idx == 1 else ck_base
            this_conv_k = target_conv_k if input_len >= target_conv_k else input_len
            conv_out_len = (input_len - this_conv_k) + 1

        if prints and this_conv_k is not None:
            print(f"  -> Conv (kernel={this_conv_k}, stride=1, pad=0)")
            print(f"  -> Length after Conv: {conv_out_len}")

        # 2. Attempt pooling search using conv_out_len
        found_p = None
        pool_output_len = None
        max_legal_p = pk_base // 2
        for p in range(max_legal_p + 1):
            l_out = math.ceil((conv_out_len + 2 * p - pk_base) / pk_s) + 1
            if l_out > 0:
                if (l_out - 1) * pk_s < (conv_out_len + p):
                    found_p = p
                    pool_output_len = l_out
                    break

        # 3. If pooling is not possible, override conv kernel to input_len and stop
        if found_p is None or pk_base is None or pk_base <= 1 or conv_out_len < pk_base or (pool_output_len is not None and pool_output_len < 1):
            if this_conv_k is not None:
                this_conv_k = input_len
            if prints and this_conv_k is not None:
                print(f"  -> Global Conv triggered! (kernel={this_conv_k})")
            ck_bases.append(this_conv_k)
            pk_bases.append(None)
            pk_ss.append(None)
            pool_pads.append(None)
            if prints: print("  -> Sequence reached length 1 or global conv. Stopping architecture build.")
            current_len = 1
            break
        else:
            ck_bases.append(this_conv_k)
            pk_bases.append(pk_base)
            pk_ss.append(pk_s)
            pool_pads.append(found_p)
            current_len = pool_output_len
            if prints:
                print(f"  -> Pool (kernel={pk_base}, stride={pk_s}, pad={found_p})")
                print(f"  -> Length after Pool: {current_len}")

        module_idx += 1

    return ck_bases, pk_bases, pk_ss, pool_pads


def Prong_W(dim_i = (1, 500, 4), dim_f = (300, 1, 1),
            
            cf_i = None, cf_pu = None,
            cf_ns = 1,

            ck_i=11, ck_base=3, 
            pool_func = nn.MaxPool2d, pk_base=4, pk_s=2, 
            actb4pool = True,
            
            activations = nn.ReLU(),
            batchnorm = 'before', bias = True,
            skip_first_conv = False,
            ): 
    
    # pk_base is an number that is added on to pk_s to determine pooling kernel size

    if ck_i is None: ck_i = ck_base
    
    c_ks, p_ks, p_ss, p_ps = Prong_W_params(dim_i[1], ck_i=ck_i, ck_base=ck_base, 
                                               pk_base=pk_base, pk_s=pk_s, prints=False, 
                                               skip_first_conv=skip_first_conv)
        
    dx = dim_i[2]

    #-------------------------------

    mods = len(c_ks)

    if cf_pu == None: cf_pu = 0
    mx, dfx = (mods-1, cf_pu) if (cf_pu > 0 and mods > 2) else (mods, dim_f[0])

    cfs = np.round(se.GeomNumSpacing(dim_i[0], dfx, mx + 1, cf_ns)).astype(int)
    
    if cf_i != None: 
        cfs = np.hstack([dim_i[0], 
                    np.round(se.GeomNumSpacing(cf_i, dfx, mx, cf_ns)).astype(int)])
    
    if cf_pu > 0: cfs = cfs.tolist() + [dim_f[0]]

    if skip_first_conv and mods > 1: cfs[1] = cfs[0] 

    #-----------------------------
    i = 0 
    layers = []
    for c_k, p_k, p_s, p_p in zip(c_ks, p_ks, p_ss, p_ps):

        if c_k is not None: 
            conv_layer = nn.Conv2d(cfs[i], cfs[i+1], kernel_size = (c_k, dx), bias = bias,
                                stride = (1, 1), padding = 0)
            layers.append(conv_layer)
            dx = 1
            skipo = False
        
        else: skipo = True

        lays_ex = []

        if p_k is not None: 

            if actb4pool and skipo is False: 
                if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
                if activations != None: lays_ex.append(activations)
                if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))

        
            pool_layer = pool_func(kernel_size = (p_k, 1), stride = (p_s, 1),
                            padding = p_p, ceil_mode = True)
            lays_ex.append(pool_layer)
        
            if actb4pool == False and skipo is False: 
                if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
                if activations != None: lays_ex.append(activations)
                if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        
        else:
            if batchnorm == 'before': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
            if activations != None: lays_ex.append(activations)
            if batchnorm == 'after': lays_ex.append(nn.BatchNorm2d(cfs[i+1]))
        
        layers.extend(lays_ex)

        i += 1 
    
    return layers














