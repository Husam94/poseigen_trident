import os
import pickle
import copy
import numpy as np
import scipy

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex
import poseigen_binmeths as bm
import poseigen_compass as co

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import time
import random

_trident_seed_counter = 0

def _set_trident_seed(seed=None):
    """
    If seed is None, generate a different seed each call.
    Returns the seed used.
    """
    global _trident_seed_counter
    _trident_seed_counter += 1

    if seed is None:
        # Unique per call within process, even for rapid consecutive calls.
        seed = (time.time_ns() + os.getpid() + _trident_seed_counter) % (2**32)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return int(seed)


#------------------------------------------

def init_mod(m, func = nn.init.kaiming_uniform_, non_linearity = 'relu'):
    
    kiamings = [nn.init.kaiming_uniform_, nn.init.kaiming_normal_]
    xaviers = [nn.init.xavier_uniform_, nn.init.xavier_normal_]
    
    if isinstance(m, nn.Conv2d):
        if func in kiamings: func(m.weight, nonlinearity=non_linearity)
        elif func in xaviers: func(m.weight, gain=nn.init.calculate_gain(non_linearity))
        else: func(m.weight)

    return

#------------------------------------------

###################################################################################################

def FindKernelSize(nums): 
    return [nums[i] - nums[i+1] + 1 for i in range(len(nums) - 1)]

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def LoadTorch(pathname): 
    if pathname.endswith('.pt') != True: pathname = pathname + '.pt'
    return torch.load(pathname, weights_only = False)

def ModelFunctioning(model, dim_i, dim_o, 
                     multi_in = False):
           
    if multi_in == False: dim_i = [dim_i]
    randos = [torch.rand(*si) for si in dim_i]
    
    try: 
        out = model(*randos)
        reto = True if out.shape == dim_o else False
    except:
        reto = False
        print('Bottleneck')
        
    return reto

###################################################################################################

class ReflectLayer(nn.Module): 
    def __init__(self, dims = -2):
        super().__init__()
        if dims == True: dims = [-2]
        self.ds = dims if isinstance(dims, list) else [dims]
    def forward(self, x):
        return torch.cat([x, torch.flip(x, dims = self.ds)], axis = -1)

class WWPLayer(nn.Module): 
    def __init__(self, func = nn.MaxPool2d):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func((1,2))(x)

class FlipLayer(nn.Module):

    #ONLY WORKS FOR TRIDENT FORMAT: (Num, Filters, LENGTH, WIDTH)

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.stack([x[:,:,:,0], torch.flip(x[:,:,:,1], [-1])], axis = -1)

##############################################

class PseudoReLU(nn.Module): 
    def __init__(self, pseudo = 0):
        super().__init__()
        self.pseudo = pseudo
        self.ReLU = nn.ReLU()
    def forward(self, x):
        return self.ReLU(x) + self.pseudo

class ExpAct(nn.Module): 
    def __init__(self, relit = False, pseudo = 0):
        super().__init__()
        self.e = 2.718281828459045

        self.psu = pseudo

        if relit: 
            self.relf = nn.ReLU()
            self.relv = 1
        
        else: 
            self.relf = nn.Identity()
            self.relv = 0

    def forward(self, x):
        return ((self.e ** (self.relf(x))) - self.relv) + self.psu

class SubtractLayer(nn.Module): 
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[:, [0]] - x[:, [1]]

class DivideLayer(nn.Module): 
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[:, [0]] / x[:, [1]]
    

#************************************************

class AsinhAct(nn.Module): 
    def __init__(self, pseudo = 0):
        super().__init__()
        self.pseudo = pseudo
    
    def forward(self, x):
        return torch.asinh(x) + self.pseudo
    

class AsinhAct_L(nn.Module):
    def __init__(self, m_init=1.0, s_init=0.0):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(m_init, dtype=torch.float))
        self.s = nn.Parameter(torch.tensor(s_init, dtype=torch.float))

    def forward(self, x):
        return self.m * torch.asinh(x) + self.s

class LeakyAsinhAct(nn.Module):
    def __init__(self, negative_slope = 0.01, pseudo = 0):
        super().__init__()
        self.negative_slope = negative_slope
        self.pseudo = pseudo
    
    def forward(self, x):
        return torch.where(x >= 0, torch.asinh(x), self.negative_slope * x) + self.pseudo
    
#********************************************************

###################################################################################################

#### TRAINER ####

def trident_loss_mode_default(): return [mex.AError, {'expo': 2}]

def norunmet_maker(Ls, metrics_mode, 
                   bino = False): 
    #bino == a boolean 
    
    Ls_stacked = [np.vstack([l[i] for l in Ls]) for i in range(len(Ls[0]))]
    if bino == False: 
        yhat, yb, w = Ls_stacked
        mm_args = metrics_mode[1]
    else: 
        yhat, yb, w, binz = Ls_stacked
        mm_args = {'bind': binz, **metrics_mode[1]}

    if isinstance(Ls[0][2], int): w = 1 # W == IN THE 3rd HERE
    
    return metrics_mode[0](yhat, yb, weights = w, **mm_args) 

def BatchFlipper(Xs, flips): 
    return [torch.flip(X, dims = F) if F != None else X for X, F in zip(Xs, flips)]


def IndivFlipper(Xs, flips, indivflip): 
    #indivflip == a list of 1 or -1 to indicate whether to flip it or not
    
    for X, F in zip(Xs, flips):
        if F != None: 
            X[indivflip == -1] = torch.flip(X[indivflip == -1], dims = F)

    return Xs


def BatchMaker(trange, trem, tspl, batchsize,
               falip, sepflips, indivflips,
               WBG, harp, tweights, tbatchs):


    fidx = None

    if WBG == None:
        scramble = np.hstack([np.random.permutation(trange), 
                              np.random.choice(trange, trem, replace = True)]).reshape(-1, batchsize)
            
    else: 

        if harp:
            scramble = bm.Harpoon(tweights, select = batchsize, 
                               multi = False, # NEEDS TO BE A FLAT LIST ALWAYS. 
                               repeat = tbatchs, custidx = None)
        else: 
            scramble = np.stack([np.random.choice(trange, size = batchsize, replace = False, p = tweights) 
                                    for _ in range(tbatchs)], 0)
            
    tidx = [tspl[h] for h in scramble]
            
    if indivflips: 
        
        fidx = [falip[h] for h in scramble]
    
    if sepflips:

        tidx = tidx * 2

        batchscram = np.random.permutation(np.arange(tbatchs * 2))

        tidx = [tidx[bs] for bs in batchscram]

        fidx = [falip[bs] for bs in batchscram]
    
    return tidx, fidx


def EpochUndersampler(obs_weight, EUS, Split, indivflips, batchsize, tbatch_prop):

    subs_mode = [se.SubSample_Random, {'weights': True}]

    if obs_weight is not None:
        obs_weight = np.array(obs_weight).reshape(-1) #just in case
        if isinstance(obs_weight[0], np.integer): 
            subs_mode = [se.SubSample_Select, {'select_mode': [bm.Harpoon, {}]}]
        
        else: 
            subs_mode = [se.SubSample_Random, {'weights': obs_weight}]

    else: 
        obs_weight = np.ones(len(np.hstack(Split)))
        
    
    new_split = bm.SubSplitGen(obs_weight, Split, onlyfirst = True, proportion = EUS, 
                                subsample_mode = subs_mode)
    
    new_tspl = new_split[0]
    if indivflips: new_tspl = np.hstack([new_tspl, new_tspl])
    new_tlength = len(new_tspl)
    new_trem = batchsize - (new_tlength % batchsize)
    
    new_trange = np.arange(new_tlength)
    new_tbatchs = int(((new_tlength + new_trem) // batchsize) * tbatch_prop)

    new_xof = new_tlength // 2 if indivflips else new_tbatchs
    new_falip = np.hstack([np.repeat(1, new_xof), np.repeat(-1, new_xof)])

    return [new_split, new_tspl, new_trem, new_trange, new_tbatchs, new_falip]




###########################################################################


def TridentTrainer(
    
    model, 
    inps, out, 
    out_std = None, out_weights = None, 
    out_bind = None,
    Split = None, 
    
    EUS = None, obs_weight = None, 
    weights_mode = None, weights_bind = True,
    WBG = None, harp = False,

    mod_init_mode = None, duds = 0, poors = None,

    dtypo = torch.float,

    flips = None, indivflips = False,
    loss_mode = trident_loss_mode_default(), loss_bind = False, 
    metrics_mode = None, smallest = None, trainmetrics = False,
    tbatch_prop = 1.0, 
    batchsize = 128, batchsize_infer = None, opt = Adam, learningrate = 0.001, maxepochs = 20, patience = 5, 
    pathname = None, statusprints = True, returnmodel = False, pickup = False,
    savebytrain = False,  seed = None
    ): 

    seed_used = _set_trident_seed(seed)

    '''
    Added batchsize_inter so that the batchsizes for inference can be different than training      
    '''

    if batchsize_infer is None: batchsize_infer = batchsize
    
    # THIS IS VERSION 2 OF THE TRAINER_BASIC!!!!!!!!!!!!!!!!!!!!!!!!!!! 

    #25-04-30 MODIFICAITON: 
        # duds_mode is now mod_init_mode.
        # Use mod_init_mode to initialize model. 

        # ADDED POORS: a list of [EPOCH THRESHOLD, PERFORM THRESHOLD]
            # Checks the peformance of every epoch aftger epoch threhsold. 
            # If its bad, it terminates. 

    ########################################################################################

    # FOR RIGHT NOW, THERE ARE NO TRAINING METRICS AND NO RUNNING METRICS. 
    # I COULD DO TRAINMETRICS BUT NO RUNNING METRICS. 

    # ADDING DUDS WHERE YOU HAVE DUD NUMBER OF TRIES TO RESET PARAMS. 

    if statusprints == True: statusprints = 1
    if statusprints == False: statusprints = None

    runningmetrics = False
    collectpredictions = False

    ########################################################################################


    if isinstance(inps, list) is False: inps = [inps]

    if metrics_mode == None: metrics_mode = loss_mode
    if smallest == None: smallest = se.metrics_smallest[metrics_mode[0]]
    
    metrics = {'Train': [], 'Validation': []}
    counter = 0
    e = 0
    
    ##########################################
    
    pn = pathname
    if pathname == None: 
        pn = 'Temp_TB_' + str(np.random.randint(100000, 999999))
        pickup = False
    
    pnMo, pnMe = pn + '_Mod.pt', pn + '_Met.p'

    if pickup and os.path.isfile(pnMe):
        metrics = pickle.load(open(pnMe, 'rb'))
        model = LoadTorch(pnMo)
        f = np.nanargmin if smallest else np.nanargmax
        bestat = f(metrics['Validation'])
        counter = len(metrics['Validation']) - bestat - 1
        best = metrics['Validation'][bestat]
        
        e = len(metrics['Validation']) - 1
    

    elif mod_init_mode is not None:
        print('initializing model')
        model = mod_init_mode[0](model, **mod_init_mode[1])

    ##########################################
    
    torch.set_printoptions(precision=6)
    
    model = model.to(device)
    optimizer = opt(model.parameters(), lr = learningrate)

    if isinstance(flips, list) and len(flips) < len(inps):  
        if len(flips) < len(inps): 
            flips = flips + [None] * (len(inps) - len(flips))

    sepflips = False
    if flips == None: 
        indivflips = False
    elif indivflips == False: sepflips = True

    epo_v = 2 if flips != None else 1

    tspl, vspl = Split[0], Split[1]
    
    if indivflips:
        tspl = np.hstack([tspl, tspl]) #now we doubled it by itself        

    tlength,vlength = (len(x) for x in [tspl, vspl])
    trem = batchsize - (tlength % batchsize)
    glength = tlength + trem
    t_ns, v_ns = (np.append(np.arange(0, y, batchsize_infer),y) for y in [tlength,vlength])

    trange = np.arange(tlength)
    tbatchs = int(((tlength + trem) // batchsize) * tbatch_prop)

    xof = tlength // 2 if indivflips else tbatchs
    falip = np.hstack([np.repeat(1, xof), 
                       np.repeat(-1, xof)])
    
    if WBG != None: 
        collectpredictions = False
        tweights = np.array(WBG)[Split[0]]
        if indivflips: tweights = np.hstack([tweights, tweights])
        if harp == False: tweights = tweights / np.sum(tweights) #now we doing weighted batch gen
    else: tweights = None
    
    ##########################################
    
    if trainmetrics != True: collectpredictions = False
    
    if torch.is_tensor(inps[0]) == False: 

        inps = [torch.from_numpy(d) for d in inps]
        out = torch.from_numpy(out)
        if out_std is not None: out_std = torch.from_numpy(out_std)
        if out_weights is not None: out_weights = torch.from_numpy(out_weights)
        #if out_bind is not None: out_bind = torch.from_numpy(out_bind)
        
    ##########################################

    if out_bind is None: 
        loss_bind = False
        weights_bind = False

    if out_weights is not None: weights_mode = None

    tpack = [Split, tspl, trem, trange, tbatchs, falip]


    def BatchData(batchidx, inps, out, out_std, out_weights, out_bind): 
        
        inps_b = [inp[batchidx] for inp in inps]
        outers = []
        for outx in [out, out_std, out_weights, out_bind]: 
            if outx is not None: outers.append(outx[batchidx])
            else: outers.append(None)
            
        return inps_b, *outers
    

    while counter < patience - 1 and e < maxepochs - 1: 

        if EUS != None: 
            tpack = EpochUndersampler(obs_weight, EUS, Split, indivflips, batchsize, tbatch_prop)

        zSplit, ztspl, ztrem, ztrange, ztbatchs, zfalip = tpack

        if weights_mode != None:
            weitarg = out_bind if weights_bind else out
        
            out_weights = weights_mode[0](weitarg, onlyidx = zSplit[0], #only the training set of the [new] split. 
                                                   **weights_mode[1])
            out_weights = torch.from_numpy(out_weights)


        tidx, fidx = BatchMaker(ztrange, ztrem, ztspl, batchsize, 
                                zfalip, sepflips, indivflips, 
                                WBG, harp, tweights, ztbatchs)

        for ib, b, in enumerate(tidx):

            inps_b, out_b, out_std_b, out_weights_b, out_bind_b = BatchData(b, inps, 
                                                                            out, out_std, out_weights, out_bind)            

            if flips != None:
                if indivflips: 
                    inps_b = IndivFlipper(inps_b, flips, fidx[ib])
                elif fidx[ib] == -1: inps_b = BatchFlipper(inps_b, flips)
            
            inps_b = [xa.to(device, dtype = dtypo) for xa in inps_b]

            out_b = out_b.to(device, dtype = dtypo)
    
            lm_args = copy.deepcopy(loss_mode[1])
            if out_weights_b is not None: lm_args.update({'weights': out_weights_b.to(device, dtype = dtypo)})
            if out_bind_b is not None and loss_bind: lm_args.update({'bind': out_bind_b})
            if out_std_b is not None: lm_args.update({'std': out_std_b.to(device, dtype = dtypo)})


            #------------------------------------------------

            optimizer.zero_grad()

            pred_b = model(*inps_b)

            loss = loss_mode[0](pred_b, out_b, **lm_args)

            loss.backward()
            optimizer.step()
            
            if ib == 0 and pred_b.shape != out_b.shape: 
                    print(f'WARNING: SHAPES DONT MATCH! Actual {out_b.shape}, Pred {pred_b.shape}')

            #------------------------------------------------
        
        if trainmetrics == False: metrics['Train'].append(np.nan)


        with torch.no_grad():
            model.eval()
            
            q = [[t_ns, 'Train', tlength, Split[0]], [v_ns, 'Validation', vlength, Split[1]]]
            a = 0 if trainmetrics and collectpredictions != True else 1
            
            for iu, u in enumerate(q[a:]): 

                metbatches = []

                for yt in range(epo_v):

                    #for metrics, you dont need to do random flipping, you do can do it one at a time
                    #also, no need to do indiv flipping either. 

                    rf = -1 if yt == 1 else 1 

                    for m in range(len(u[0])-1):
                        
                        sel0, sel1 = u[0][m], u[0][m+1]
                        b = u[3][sel0:sel1]

                        inps_b, out_b, out_std_b, out_weights_b, out_bind_b = BatchData(b, inps, out, out_std, out_weights, out_bind)

                        if flips != None and rf == -1: inps_b = BatchFlipper(inps_b, flips)
                        inps_b = [xa.to(device, dtype = dtypo) for xa in inps_b]

                        pred_b = model(*inps_b).cpu().detach()

                        metbatches.append([pred_b, out_b, out_std_b, out_weights_b, out_bind_b])
            

                pred_mbs = np.vstack([mb[0].numpy() for mb in metbatches])
                out_mbs = np.vstack([mb[1].numpy() for mb in metbatches])

                mm_args = copy.deepcopy(metrics_mode[1])


                if metbatches[0][2] is not None: mm_args.update({'std': np.vstack([mb[2].numpy() for mb in metbatches])})
                if metbatches[0][3] is not None: mm_args.update({'weights': np.vstack([mb[3].numpy() for mb in metbatches])})
                if metbatches[0][4] is not None: mm_args.update({'bind': np.vstack([mb[4] for mb in metbatches])})

                metrics[u[1]].append(metrics_mode[0](pred_mbs, out_mbs, **mm_args))


        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 

        pickle.dump(metrics, open(pnMe, 'wb'))

        if e <= 1: best = np.nan_to_num(metrics['Validation'][e])
            
        tre = best > metrics['Validation'][e] if smallest else best < metrics['Validation'][e]
        
        if tre or e <= 1: #so now models start with a counter 0 and get saved. 
            best = metrics['Validation'][e]
            counter = 0
            torch.save(model, pnMo)


        elif (pred_mbs == pred_mbs[0, 0, 0, 0]).all():

            print('SAME OUTPUTS')

            if duds > 0 and mod_init_mode is not None: 
            
                mod_init_mode[0](model, **mod_init_mode[1])

                counter = 0

                duds -= 1

                print(f'MODEL RESET, REMAINING DUDS: {duds}')
            
            else:

                print('POOR MODEL, TERMINATED')
                counter = patience
                break

        elif metrics['Validation'][e] == metrics['Validation'][e-1]: 

            print('STUCK')

            print('POOR MODEL, TERMINATED')
            counter = patience
            break

        else: counter += 1
                   
        if statusprints is not None: 
            if e % statusprints == 0: 
                print(f"E {e+1} Training: {metrics['Train'][e]} Validation: {metrics['Validation'][e]} Counter {counter}")
        

        #++++++++++++++++++++++++++++++++++++++++++

        if savebytrain: 

            pnMo_train = pn + '_Mod_TRAIN.pt'

            best_train = np.nanmin(metrics['Train'][e]) if smallest else np.nanmax(metrics['Train'][e])
            if metrics['Train'][e] == best_train: torch.save(model, pnMo_train)
        
        #******************************************

        if poors is not None: 
            if e >= poors[0] - 1:
                metx = metrics['Validation'][e]
                cuti = metx >= poors[1] if smallest else metx <= poors[1]
                if cuti: 
                    print('POOR MODEL, TERMINATED')
                    counter = patience
                    break
        
        #******************************************

        e += 1
    
    if returnmodel == True: mod = LoadTorch(pnMo)
    
    if pathname == None: os.remove(pnMo), os.remove(pnMe)
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return (metrics, mod) if returnmodel else metrics







###################################################################################################

def TridentPredictor(model, Xdatas, batchsize, flips = None, avg_flips = False):
    #Xdatas == a list of inputs for a model
    #added flips where it will add the flips to the end 

    if isinstance(model, str): model = LoadTorch(model)
        
    if isinstance(Xdatas, list) == False: Xdatas = [Xdatas]
    if torch.is_tensor(Xdatas[0]) == False: Xdatas = [torch.from_numpy(d) for d in Xdatas]

    model = model.to(device) #may need work here

    epo_p = 2 if flips != None else 1

    plength = len(Xdatas[0])
    p_ns = np.append(np.arange(0, plength, batchsize),plength)
    
    preds = []
    with torch.no_grad():
        model.eval()

        for yt in range(epo_p):

            rf = -1 if yt == 1 else 1 

            for b in range(len(p_ns)-1): 
                
                sel0, sel1 = p_ns[b], p_ns[b+1]
                
                Xs = [d[sel0:sel1] for d in Xdatas]

                if flips != None and rf == -1: Xs = BatchFlipper(Xs, flips)

                Xs = [xa.to(device, dtype = torch.float) for xa in Xs]
                
                preds.append(model(*Xs).cpu().detach().numpy())
            
    
    preds = np.vstack(preds)
    if flips != None and avg_flips: 
        lendiv = len(preds) // 2
        preds1, preds2 = preds[:lendiv], preds[lendiv:]
        preds = (preds1 + preds2) / 2
    
    return preds


def WeightedPreds(folder, weights, repeats = 10, ext = '_Preds'):
    preds_all = np.array([se.PickleLoad(folder + f'0_{x}' + ext) for x in range(repeats)])
    weights_rs = np.array(weights)[:repeats].reshape(-1, *([1] * (len(preds_all.shape)-1)))
    preds_weighted = np.sum(preds_all * weights_rs, axis = 0) / np.sum(weights)
    return preds_weighted


###################################################################################################

def TridentPredMulti(mod, Xs, 
                     flips = None, avg_flips = False, 
                     batchsize = 128, pn_preds = None,
                     rewrite = False): 
    

    if mod is not None: 
        if isinstance(mod, list) == False: mod = [mod]
    if pn_preds is not None: 
        if isinstance(pn_preds, list) == False: pn_preds = [pn_preds]

        preds = []
        for mo, pp in zip(mod, pn_preds): 
            if os.path.isfile(pp) and rewrite is False:
                predx = se.PickleLoad(pp)
            else: 
                predx = TridentPredictor(mo, Xs, flips = flips, avg_flips=avg_flips,
                                          batchsize = batchsize)
                se.PickleDump(predx, pp)
            
            preds.append(predx)
        
    else:
        preds = [TridentPredictor(mo, Xs, flips = flips, avg_flips=avg_flips,
                                  batchsize = batchsize) for mo in mod]
    
    if len(preds) == 1: preds = preds[0]
    else: 
        preds = np.stack(preds, axis = 0).mean(axis = 0)
    
    return preds


def FindSmallest_Trident(trainer_args): 
    #trainer_args needs to have always has 'loss_mode' and 'metrics_mode' 
    
    m = 'metrics_mode'
    if 'metrics_mode' not in trainer_args.keys():
        m = 'loss_mode'
    elif trainer_args['metrics_mode'] is None: m = 'loss_mode'
    
    mode = None if m == 'loss_mode' and 'loss_mode' not in trainer_args.keys() else trainer_args[m]
    if mode is None: mode = trident_loss_mode_default
    
    return mode[0]

##############################################


TCS_args = {'trainer': TridentTrainer, 'trainer_args': {}, 'smallest': None,
            'pathname': None, 'returnmodel': False,
            'get_predictions': True, 'pred_rewrite': False, 'add_pred_args': {}, 
            'metrics_mode': None, 'use_sampleweights': False,
            'score_on': 2}

def TridentCanScorer(algo, algo_args,
                     data = None, Split = None,
                     trainer = TridentTrainer, trainer_args = {}, smallest = None,
                     pathname = None, returnmodel = False,

                     get_predictions = True, pred_rewrite = False, add_pred_args = {}, 
                     metrics_mode = None, use_sampleweights = False,
                     score_predictions = False,
                     score_on = 2, score_only = False): 
    
    
    #Trident Trainers return a dictionary of metrics with the following structure: 
    # metrics: {'Train': [n,n,n,], 'Validation': [n,n,n,n,]}
    #Training hyperparameters: learning rate, optimizer, batchsize 
    
    #metrics_mode removed, needs to be in trainer_args... 

    #25.01.15 modifications: 
        # get_predictions: gets the predictions on all the data. Must have savemodels enabled and a pathname
        # score_on (IF GET_PREDICTIONS): scores the model based on this idx of the split (0 = 1)

    #25.02.03 modifications: 

        # DATA AND SPLIT COULD BE NONE - ALREADY SPECIFIED IN TRAINER ARGS. 
        # Also NO USE SAMPLE WEIGHTS. 

        #If score_only, then ONLY do predicitons for the score_on

    #-------------------------------------------

    if Split is not None: trainer_args['Split'] = Split
    if data is not None: trainer_args['data'] = data

    #-------------------------------------------

    if metrics_mode is None: metrics_mode = trident_loss_mode_default()

    trainer_args_prefix = 'TA_'
    
    algo_args_copy = algo_args.copy()
    traininghp = ['batchsize', 'opt', 'learningrate']
    aak = list(algo_args_copy.keys())

    for a in aak: 
        if a in traininghp: 
            trainer_args[a] = algo_args_copy[a]
            del algo_args_copy[a]
        elif a.startswith(trainer_args_prefix): 
            a2 = a.removeprefix(trainer_args_prefix)
            trainer_args[a2] = algo_args_copy[a]
            del algo_args_copy[a]

    m = algo(**algo_args_copy)
    
    if smallest is None: smallest = FindSmallest_Trident(trainer_args)
            
    trainer_args['pathname'] = pathname
    trainer_args['returnmodel'] = returnmodel

    trainerout = trainer(m, **trainer_args)


    r = np.nanmin if smallest else np.nanmax 
    met = r(trainerout['Validation']) if returnmodel is False else r(trainerout[0]['Validation'])






    if 'metrics_mode' in trainer_args: metrics_mode = trainer_args['metrics_mode']
    elif 'loss_mode' in trainer_args: metrics_mode = trainer_args['loss_mode']


    if get_predictions: 

        inps = trainer_args['inps']

        if 'batchsize' not in list(add_pred_args.keys()): 
            if 'batchsize' in aak: add_pred_args['batchsize'] = trainer_args['batchsize']
            else: add_pred_args['batchsize'] = 128 # DEFAULT

        if 'flips' not in list(add_pred_args.keys()): 
            if 'flips' in trainer_args: add_pred_args['flips'] = trainer_args['flips']

        add_pred_args['rewrite'] = pred_rewrite # TAKES PRECENDENCE

        idx_sel = trainer_args['Split'][score_on]
        
        ext, inps_predo = '', inps
        if score_only: 
            ext = '_only_' + str(score_on)
            inps_predo = [qer[idx_sel] for qer in inps]

        preds = TridentPredMulti(pathname + '_Mod.pt', inps_predo, 
                                 pn_preds = pathname + '_Preds' + ext + '.p', 
                                 **add_pred_args)


        if score_predictions: 

            for keyo in ['out_std', 'out_weights', 'out_bind']: 
                if keyo not in trainer_args.keys(): trainer_args[keyo] = None

            out, out_std, out_weights, out_bind = [trainer_args[x] for x in ['out', 'out_std', 'out_weights', 'out_bind']]
            
            out_sel = out[idx_sel]
            preds_sel = preds if score_only else preds[idx_sel]

            mm_args = copy.deepcopy(metrics_mode[1])
            if out_std is not None: mm_args.update({'std': out_std[idx_sel]})
            if out_bind is not None: mm_args.update({'bind': out_bind[idx_sel]})

            met = metrics_mode[0](preds_sel, out_sel, **mm_args)

    
    else:
        print('HHHHHHHHHHHHHHHHHHHHHHHHHHH')

    return (met, trainerout[1]) if returnmodel else met


TCR_args = {'Splits': None, 'repeats': 3,
            
            'trainer': TridentTrainer, 'trainer_args': {}, 'smallest': None,
            'get_predictions': True, 'pred_rewrite': False, 'add_pred_args': {}, 
            'metrics_mode': None,
            'score_on': 2, 
            
            'pickup': False, 'statusprints': True, 'pathname': None, 'savemodels': False, 
            'ext': None, 'returnmodel': False} 

def TridentCanRepeater(algo, algo_args, data, Splits = None, 
                       repeats = 3, 
                       
                    #TRIDENTCANSCORER ARGS: 
                       trainer = TridentTrainer, trainer_args = {}, smallest = None,
                       get_predictions = True, pred_rewrite = False, add_pred_args = {}, 
                       metrics_mode = None,
                       score_on = 2, score_only = False, score_predictions = False,

                       pickup = False, statusprints = True, pathname = None, savemodels = False, 
                       ext = None, returnmodel = False):

    #SEPT 20 2023 EDIT: MUST BE DATA + SPLIT FORMAT, NOT PRE-SPLIT GARBO. 

    if smallest is None: smallest = FindSmallest_Trident(trainer_args)

    if ext is True: ext = 'TriCanRep'
    newpathname = se.NewFolder(pathname, ext = ext)

    if Splits is None: Splits = [None]    
    elif isinstance(Splits[0], list) is False: Splits = [Splits]
    lsp = len(Splits)
            
    pnTR_Me = newpathname + 'Mets.p'
    
    modelcombos = []
    newmetrics = {} 
    for s in range(lsp):
        newmetrics[s] = {}
        for r in range(repeats):
            newmetrics[s][r] = None
            modelcombos.append([s, r])

    if pickup and os.path.isfile(pnTR_Me): 
        oldmetrics = se.PickleLoad(pnTR_Me)
        for s in oldmetrics.keys(): 
            for r in oldmetrics[s].keys(): 
                newmetrics[s][r] = oldmetrics[s][r]

    savepath = None

    TCS_args = {'trainer': trainer, 'trainer_args': trainer_args, 
                'smallest': smallest, 'returnmodel': False, 'metrics_mode': metrics_mode,
                'get_predictions': get_predictions, 'pred_rewrite': pred_rewrite, 
                'score_on': score_on, 'score_only': score_only, 'score_predictions': score_predictions,
                'add_pred_args': add_pred_args}
    
    for c in modelcombos: 
        s, r = c
        
        if newmetrics[s][r] is None:
           
            if statusprints == True: print(f' Cross val {s+1} of {lsp}, Repeat {r+1} of {repeats}')

            if newpathname is not None and savemodels == True: savepath = newpathname + str(s) + '_' + str(r)

            newmetrics[s][r] = TridentCanScorer(algo, algo_args, data, Split = Splits[s],
                                                **TCS_args, pathname = savepath)

            if statusprints == True: print(f"Score of {s, r}: {newmetrics[s][r]}")

            if newpathname is not None: pickle.dump(newmetrics, open(pnTR_Me, 'wb'))

    metrics = co.Metrics2Flat({0: se.PickleLoad(pnTR_Me)})

    metrics = np.array([m for m in metrics if m is not None])
    
    f = np.nanargmin if smallest else np.nanargmax
    bm = f(metrics)
    bz = metrics[bm]
    bs, br = modelcombos[bm]
        
    print(f'Best Score {bz} at Cross val {bs+1} of {lsp} and Repeat {br+1} of {repeats}')
    
    if newpathname is not None: 
        bestpath = newpathname + str(bs) + '_' + str(br) + '_Mod.pt'
        print(f'The pathname is: {bestpath}')
    
        if returnmodel: 
            bestmod = LoadTorch(bestpath)
            bestmod = bestmod.eval()
    else: returnmodel = False
    
    return (metrics, bestmod) if returnmodel else metrics






##################################









##################################


def ByAxis(inp, byaxis = -1, mode = None, pyt = False): 
    #applies a function by axis
    
    origshape = inp.shape
    lenba = origshape[byaxis]

    xp = torch if isinstance(inp, torch.Tensor) else np

    inps = xp.split(inp, lenba, byaxis)

    if mode is not None: inps = [mode[0](np, **mode[1]) for np in inps]

    return inps





def BinnedLoss(inp1, inp2, std = None, 
               
               weights = None, useweights = True, 

                bind = None, bin_sizes = None,
                byaxis = None, seperate = False,

                metrics_mode = [mex.AError, {'expo': 2}],
                summarize_mode = [mex.MeanExpo, {'expo': 2}]):
    

    ''' 
    Added bin_sizes to inform of the sizes of the bins as proportion of the total range.
    Needs to be of shape (inp1.shape[byaxis], num_bins) and must add to 1.
    
    '''

    xp = torch if isinstance(inp1, torch.Tensor) else np
    
    #uni is determined by the input always. 
    
    weio = True if weights is not None and isinstance(weights, int) is False and useweights else False

    inp = [inp1, inp2] if std is None else [inp1, inp2, std]
    if weio: inp = [*inp, weights]

    if bin_sizes is None: bin_sizes = [None] * 10

    if byaxis is not None:
        inp_ba = [ByAxis(p, byaxis = byaxis) 
                  for p in inp] #so now a list of lists, each list having seperated by axis 
        inp_pairs = [[p[i] for p in inp_ba] for i in range(len(inp_ba[0]))]
        bind = ByAxis(bind, byaxis = byaxis) #BY DEFAULT IT IS NUMPY
        
    else: 
        inp_pairs = [inp]
        bind = [bind]
        

    uni =  [np.unique(b) for b in bind]

    gxs = []
    
    for ip, bi, un, bxo in zip(inp_pairs, bind, uni, bin_sizes):

        par = bm.BinParser(bi.reshape(-1), uni = un, categorical = True)

        gx = []

        for a in par: 

            if len(a) == 0: continue 


            weighto = ip[-1][a] if weio else None

            sco = metrics_mode[0](*[b[a] for b in ip[:-1]], weights = weighto, 
                                    **metrics_mode[1])
            
            gx.append(sco)

        
        gx = xp.stack(gx)

        if bxo is not None: 

            bxo = [x for x, a in zip(bxo, par) if len(a) > 0]

            # Ensure bxo matches inp1 type and device
            if torch.is_tensor(inp1):
                bxo = torch.tensor(bxo, dtype=inp1.dtype, device=inp1.device)
            else:
                bxo = np.array(bxo, dtype=inp1.dtype if hasattr(inp1, 'dtype') else None)
            
            gx = gx * bxo
        
        gxs.append(gx)
    
    if summarize_mode is not None: 
    
        if seperate: 
            gx_sums = [summarize_mode[0](gx, **summarize_mode[1]) for gx in gxs]
        else: 
            gxs = xp.stack(gxs)
            gx_sums = summarize_mode[0](gxs, **summarize_mode[1])
    
    else: gx_sums = gxs

    return gx_sums




#################################################

def SynthDataGen(model,
                 num_obs = 100000,
                 shape_in = (10, 1, 1),
                 shape_out = (1, 1, 1), 
                 bounds = (-1, 1),
                 noise = 1, #means 0.5 standard deviation bounds for noise sampling
                 batchsize = 256): 
        
    X = np.random.uniform(bounds[0], bounds[1], size = (num_obs, *shape_in))

    #X = np.random.rand(num_obs, *shape_in)
    Y = TridentPredictor(model, X, batchsize = batchsize).reshape((num_obs, *shape_out))
    #plt.hist(Y.reshape(-1), alpha = 0.5)

    if noise is not None: 
        mean,std = (f(Y) for f in (np.mean, np.std))
        print(mean, std)
        noise = scipy.stats.truncnorm.rvs(a = -noise, b = noise, loc = 0, scale = std, 
                                          size = (num_obs, *shape_out))
        
        #plt.hist(noise.reshape(-1), alpha = 0.5)
        #plt.scatter(*[x.reshape(-1)[::10] for x in [Y, Y+noise]])

        Y = Y+noise

        #plt.hist(Y.reshape(-1), alpha = 0.5)

    return X,Y







def FeatExtract(model, layer_name, inp, batchsize = 256): 
    #from pytorch forums 

    if isinstance(model, str): model = LoadTorch(model)

    model.to(device)
    model = model.eval()

    if isinstance(inp, list) is False: inp = [inp]
    
    if torch.is_tensor(inp[0]) == False: 
        inp = [torch.from_numpy(ix) for ix in inp]

    rem = len(inp[0]) % batchsize
    num_batches = (len(inp[0]) // batchsize) + rem

    activation = {}
    def get_activation(layer_name):
        def hook(model, input, output):
            activation[layer_name] = output.detach().cpu().numpy()
        return hook
    
    alls = []

    for nb in np.arange(num_batches): 
        first = nb * batchsize 
        batch = [ix[first: first + batchsize].to(device, dtype = torch.float) for ix in inp]

        getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
        _ = model(*batch)
        alls.append(activation[layer_name])
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return np.vstack(alls)




def TridentWindow(inp, size): 

    ''' returns shape of (N, num_win, num signals, length, width) '''
    ST = np.lib.stride_tricks.sliding_window_view
    winshape = (inp.shape[1], size, inp.shape[-1])
    sig_win = ST(inp, (1, *winshape))
    sig_win_rs = sig_win[:, 0, :, 0, 0]
    return sig_win_rs












#=====================================================================


def BinnedLoss_Overlap(inp1, inp2, std=None,
                  weights=None, useweights=True,
                  bind=None,
                  byaxis=None, seperate=False,
                  metrics_mode=[mex.AError, {'expo': 2}],
                  summarize_mode=[mex.MeanExpo, {'expo': 2}],
                  pyt=False,indiv_error = False):
    

    ''' bind needs to be of shape (N, maxbins) or if byaxis then an additional dimension there'''

    weio = weights is not None and not isinstance(weights, int) and useweights

    inp = [inp1, inp2] if std is None else [inp1, inp2, std]
    if weio:
        inp = [*inp, weights]

    if byaxis is not None:
        inp_ba = [ByAxis(p, byaxis=byaxis, pyt=pyt) for p in inp]
        inp_pairs = [[p[i] for p in inp_ba] for i in range(len(inp_ba[0]))]
        bind = ByAxis(bind, byaxis=byaxis, pyt=False)
    else:
        inp_pairs = [inp]
        bind = [bind]

    gxs = []

    indiv_errors = []


    for ip, bi in zip(inp_pairs, bind):
    
        par = se.Belong2Window(bi)

        if weio:
            gx = [metrics_mode[0](*[b[a] for b in ip[:-1]], weights = ip[-1][a], 
                                  **metrics_mode[1])
                                  for a in par if len(a) > 1]
        else: 
            gx = [metrics_mode[0](*[b[a] for b in ip], 
                                  **metrics_mode[1])
                                  for a in par if len(a) > 1]
        
        gx = torch.stack(gx) if pyt else np.array(gx)
        
        gxs.append(gx)


        if indiv_error: 
            
            mask1 = np.full(bi.shape, np.nan, dtype=np.float32)
            gx_np = gx.cpu().numpy() if pyt else gx

            # Use unique bins to avoid out-of-bounds and ensure all bins are covered
            unique_bins = np.unique(bi)
            for bix in unique_bins:
                mask1[bi == bix] = gx_np[bix]

            indiv_err = np.nanmean(mask1, axis=1)
            indiv_errors.append(indiv_err)
            
    
    if summarize_mode is not None:
        if seperate:
            gx_sums = [summarize_mode[0](gx, **summarize_mode[1]) for gx in gxs]
        else:
            gxs = torch.stack(gxs) if pyt else np.stack(gxs)
            gx_sums = summarize_mode[0](gxs, **summarize_mode[1])
    else:
        gx_sums = gxs

    if indiv_error:
        return gx_sums, indiv_errors if len(indiv_errors) > 1 else indiv_errors[0]
    else:
        return gx_sums








































#+=======================================================================================T

def EpochUndersampler_BW(bw_train, tspl, bind_train, bin_sizes,
                         EUS_prop, EUS_mode, boost_weights = None, 
                         counter = None, patience = None, replace = False):

    '''
    EUS_mode can be either "recip", "comp", "ebs"
    "ebs" mode only supports single value outputs currently. 

    '''

    if EUS_prop == 1 and replace is False: 
        return tspl


    num_sel = int(len(bw_train) * EUS_prop)

    #----------------------------------------------------------------

    if EUS_mode == 'PB': 
        propos = se.GeomNumSpacing(EUS_prop, 1.0, patience - 1, plier = 1)

        propos_cur = propos[counter]

        if propos_cur == 1 and replace is False: 
            return tspl

        num_sel = int(len(bw_train) * propos_cur)

        print(f'DOING {num_sel}')

        

        new_tspl = np.random.choice(tspl, num_sel, p = bw_train.reshape(-1), replace = replace)


    #----------------------------------------------------------------

    if EUS_mode == 'opt': 

        opt_samp = bm.OptSample(bind_train, bin_sizes=bin_sizes, select=EUS_prop, 
                                n_samples=100, random_state=None)
        
        new_tspl = tspl[opt_samp]
    #----------------------------------------------------------------
        
    if EUS_mode != 'ebs': 

        if boost_weights is None: boost_weights = 1

        combw_train = bw_train * boost_weights

        combw_train = np.prod(combw_train, axis=1, keepdims=True)

        combw_train = combw_train / np.sum(combw_train)

        new_tspl = np.random.choice(tspl, num_sel, p = combw_train.reshape(-1), replace = replace)
    
    else: 
        pass #Not supporting ebs anymore. 
        
        
    
    return new_tspl


def BatchMaker_BW(x, batchsize, make_full=False):
    """
    Efficiently splits a shuffled index array of length x into batches of size batchsize.
    If x is not divisible by batchsize:
      - if make_full is True, fills the last batch with random indices.
      - if make_full is False, the last batch will be smaller.
    Returns a list of numpy arrays (batches of indices).
    """
    idx = np.arange(x)
    np.random.shuffle(idx)
    n_full = x // batchsize
    n_rem = x % batchsize

    batches = [idx[i*batchsize:(i+1)*batchsize] for i in range(n_full)]
    if n_rem > 0:
        if make_full:
            fill = np.random.choice(idx, batchsize - n_rem, replace=True)
            last_batch = np.concatenate((idx[n_full*batchsize:], fill))
            batches.append(last_batch)
        else:
            batches.append(idx[n_full*batchsize:])
    return batches

def BatchMaker_BW_Torch(x, batchsize, make_full=False, device=device):
    """
    Efficiently splits a shuffled index tensor of length x into batches of size batchsize.
    If x is not divisible by batchsize:
      - if make_full is True, fills the last batch with random indices.
      - if make_full is False, the last batch will be smaller.
    Returns a list of torch tensors (batches of indices) on the specified device.
    """
    idx = torch.randperm(x, device=device)
    n_full = x // batchsize
    n_rem = x % batchsize

    batches = [idx[i*batchsize:(i+1)*batchsize] for i in range(n_full)]
    if n_rem > 0:
        if make_full:
            fill = idx[torch.randint(0, x, (batchsize - n_rem,), device=device)]
            last_batch = torch.cat((idx[n_full*batchsize:], fill))
            batches.append(last_batch)
        else:
            batches.append(idx[n_full*batchsize:])
    return batches



def BatchData_BW(batchidx, inps, out, out_std, out_weights): 
        
        inps_b = [inp[batchidx] for inp in inps]
        outers = []
        for outx in [out, out_std, out_weights]: 
            if outx is not None: outers.append(outx[batchidx])
            else: outers.append(None)
            
        return inps_b, *outers



def TridentTrainer_BW(
    
    model, 

    inps, out, 

    out_std = None, 

    out_bind = None, bin_sizes = None, bw_mode = 'recip', 

    Split = None, 

    EUS_prop = None, EUS_mode = 'recip', EUS_replace = False, boost = False,

    mod_init_mode = None, duds = 0, poors = None,

    dtypo = torch.float,

    flips = None, 

    loss_mode = trident_loss_mode_default(), 
    
    metrics_mode = None, smallest = None, trainmetrics = False,
        
    batchsize = 128, batchsize_infer = None, opt = Adam, learningrate = 0.001, maxepochs = 20, patience = 5, 
    
    pathname = None, statusprints = True, returnmodel = False, pickup = False,
    
    savebytrain = False, seed = None,
    ):

    seed_used = _set_trident_seed(seed)

    '''

    Started: 2025-12-15

    This version weighs the observations by bin weights per epoch and applies the weights to metrics that use the weights. 
    No more loss_bind since we are doing weighting instead of local error based metrics. 
    
    Epoch Sampling is either by using weights or Harpoon 
  
    '''

    if batchsize_infer is None: batchsize_infer = batchsize
    
    if statusprints == True: statusprints = 1
    if statusprints == False: statusprints = None

    collectpredictions = False

    ########################################################################################


    if isinstance(inps, list) is False: inps = [inps]

    if metrics_mode == None: metrics_mode = loss_mode
    if smallest == None: smallest = se.metrics_smallest[metrics_mode[0]]
    
    metrics = {'Train': [], 'Validation': []}
    counter = 0
    e = 0
    
    ##########################################
    
    pn = pathname
    if pathname == None: 
        pn = 'Temp_TB_' + str(np.random.randint(100000, 999999))
        pickup = False
    
    pnMo, pnMe = pn + '_Mod.pt', pn + '_Met.p'

    if pickup and os.path.isfile(pnMe):
        try:
            metrics = pickle.load(open(pnMe, 'rb'))
            model = LoadTorch(pnMo)
            f = np.nanargmin if smallest else np.nanargmax
            bestat = f(metrics['Validation'])
            counter = len(metrics['Validation']) - bestat - 1
            best = metrics['Validation'][bestat]
            e = len(metrics['Validation']) - 1
        except Exception as ex:
            print(f"failed at : {pnMe}")
            metrics = {'Train': [], 'Validation': []}
            counter = 0
            e = 0
            best = np.inf if smallest else -np.inf
            if mod_init_mode is not None:
                print('Reinitializing model due to failed load.')
                model = mod_init_mode[0](model, **mod_init_mode[1])
    elif mod_init_mode is not None:
        print('initializing model')
        model = mod_init_mode[0](model, **mod_init_mode[1])
        counter = 0
        e = 0
        best = np.inf if smallest else -np.inf

    ##########################################
    
    torch.set_printoptions(precision=6)
    
    model = model.to(device)
    optimizer = opt(model.parameters(), lr = learningrate)

    if trainmetrics != True: collectpredictions = False
    
    if torch.is_tensor(inps[0]) == False: 

        inps = [torch.from_numpy(d).to(device,  dtype = dtypo) for d in inps]
        out = torch.from_numpy(out).to(device,  dtype = dtypo)
        if out_std is not None: out_std = torch.from_numpy(out_std).to(device,  dtype = dtypo)

    #-----------------------------------------------------------------------------------------

    if isinstance(flips, list) and len(flips) < len(inps):  
        if len(flips) < len(inps): 
            flips = flips + [None] * (len(inps) - len(flips))


    epo_v = 2 if flips != None else 1  

    unique_bins = [np.unique(out_bind[:, g]) for g in np.arange(out_bind.shape[1])]

    boost_weights = None


    bw_base = {x: bm.BinWeights(out_bind[y], bin_sizes = bin_sizes, mode = bw_mode) 
               for x, y in zip(['Train', 'Validation'], [Split[0], Split[1]])}
    
    batchidx_base = {x: BatchMaker_BW_Torch(len(out[y]), batchsize = batchsize_infer, make_full=False)
                     for x, y in zip(['Train', 'Validation'], [Split[0], Split[1]])}


    while counter < patience - 1 and e < maxepochs - 1: 

        tspl = Split[0]

        if EUS_prop is not None: 

            bw_train = bw_base['Train']

        
            # if e > 0 and boost: 

            #     boost_weights = LossWeights(model, tspl, 
            #                       inps, out, out_std, 
            #                       out_bind, metrics_mode, flips, batchsize, per_bin = 20, 
            #                       dtypo = dtypo)
            
            # else: boost_weights = None
            
            tspl = EpochUndersampler_BW(bw_train, tspl, out_bind[tspl], bin_sizes,
                                        EUS_prop, EUS_mode, boost_weights = boost_weights, 
                                        counter = counter, patience = patience, 
                                        replace = EUS_replace)
        
        #----------------

        tra_bind = out_bind[tspl] 
        tra_wei = bm.BinWeights(tra_bind, bin_sizes = bin_sizes, mode = bw_mode) 
        tra_wei = torch.from_numpy(tra_wei).to(device,  dtype = dtypo)
        #-----------------
        tspl = torch.as_tensor(tspl, device=device)
        tra_out = out[tspl]
        tra_std = out_std[tspl] if out_std is not None else None
        tra_inp = [x[tspl] for x in inps]
        #-----------------------------

        if flips is not None: 
            tra_out, tra_wei = [torch.vstack([x, x]) for x in [tra_out, tra_wei]]
            if tra_std is not None: tra_std = torch.vstack([tra_std, tra_std])
            tra_inp = [torch.vstack([x, torch.flip(x, dims = F)]) for x, F in zip(tra_inp, flips)]
        
        tidx = BatchMaker_BW_Torch(len(tra_out), batchsize = batchsize, make_full=True)


        used = []

        for ib, b, in enumerate(tidx):

            tra_inp_b, tra_out_b, tra_std_b, tra_wei_b = BatchData_BW(b, tra_inp, tra_out, tra_std, tra_wei)            

            lm_args = copy.deepcopy(loss_mode[1])
            if tra_wei_b is not None: lm_args.update({'weights': tra_wei_b})
            if tra_std_b is not None: lm_args.update({'std': tra_std_b})

            #------------------------------------------------

            optimizer.zero_grad()

            pred_b = model(*tra_inp_b)

            loss = loss_mode[0](pred_b, tra_out_b, **lm_args)
            loss.backward()
            optimizer.step()
            
            if ib == 0 and pred_b.shape != tra_out_b.shape: 
                print(f'WARNING: SHAPES DONT MATCH! Actual {tra_out_b.shape}, Pred {pred_b.shape}')

            #------------------------------------------------

            if boost: 
                yex = [x for x in [tra_out_b, pred_b.detach()]]
                if out_std is not None: yex.extend(tra_std_b)
                used.append(yex)
        
        if boost: 
            used_outs, used_preds = [torch.cat([x[y] for x in used], dim = 0).cpu().numpy() for y in [0, 1]]
            if out_std is not None: used_std = torch.cat([x[2] for x in used], dim = 0).cpu().numpy()
            else: used_std = None
            used_idx = torch.hstack(tidx).cpu().numpy()
            used_bind = tra_bind[used_idx]

            boost_weights = LossWeights_Collect(used_outs, used_preds, used_std, used_bind, 
                                                out_bind[Split[0]],                             # NEED TO DO THIS
                                                metrics_mode, unique_bins)



        if trainmetrics == False: metrics['Train'].append(np.nan)


        
        with torch.no_grad():
            model.eval()
            
            q = [['Train', Split[0]], ['Validation', Split[1]]]
            a = 0 if trainmetrics and collectpredictions != True else 1
            
            for iu, u in enumerate(q[a:]): 

                ev_spl = torch.as_tensor(u[1], device=device)
                ev_out = out[ev_spl]
                ev_std = out_std[ev_spl] if out_std is not None else None
                ev_inp = [x[ev_spl] for x in inps]
                
                ev_wei = torch.from_numpy(bw_base[u[0]]).to(device,  dtype = dtypo)

                ev_idx = batchidx_base[u[0]]

                metbatches = []

                for yt in range(epo_v):

                    if yt == 1: ev_inp = [torch.flip(x, dims = F) for x, F in zip(ev_inp, flips)]

                    for ib, b in enumerate(ev_idx): 

                        ev_inp_b, ev_out_b, ev_std_b, ev_wei_b = BatchData_BW(b, ev_inp, ev_out, ev_std, ev_wei)            

                        pred_b = model(*ev_inp_b)

                        metbatches.append([pred_b, ev_out_b, ev_std_b, ev_wei_b])

                pred_mbs = torch.cat([mb[0] for mb in metbatches], dim = 0)
                out_mbs = torch.cat([mb[1] for mb in metbatches], dim = 0)

                mm_args = copy.deepcopy(metrics_mode[1])
                if ev_std is not None: 
                    std_mbs = torch.cat([mb[2] for mb in metbatches], dim = 0)
                    mm_args.update({'std': std_mbs})
                
                if ev_wei is not None: 
                    wei_mbs = torch.cat([mb[3] for mb in metbatches], dim = 0)
                    mm_args.update({'weights': wei_mbs})                

                metrics[u[0]].append(metrics_mode[0](pred_mbs, out_mbs, **mm_args).cpu().item())


        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 

        pickle.dump(metrics, open(pnMe, 'wb'))

        if e <= 1: best = np.nan_to_num(metrics['Validation'][e])
            
        tre = best > metrics['Validation'][e] if smallest else best < metrics['Validation'][e]
        
        if tre or e <= 1: #so now models start with a counter 0 and get saved. 
            best = metrics['Validation'][e]
            counter = 0
            torch.save(model, pnMo)


        elif (pred_mbs == pred_mbs[0, 0, 0, 0]).all():

            print('SAME OUTPUTS')

            if duds > 0 and mod_init_mode is not None: 
            
                mod_init_mode[0](model, **mod_init_mode[1])

                counter = 0

                duds -= 1

                print(f'MODEL RESET, REMAINING DUDS: {duds}')
            
            else:

                print('POOR MODEL, TERMINATED')
                counter = patience
                break

        elif metrics['Validation'][e] == metrics['Validation'][e-1]: 

            print('STUCK')

            print('POOR MODEL, TERMINATED')
            counter = patience
            break

        else: counter += 1
                   
        if statusprints is not None: 
            if e % statusprints == 0: 
                print(f"E {e+1} Training: {metrics['Train'][e]} Validation: {metrics['Validation'][e]} Counter {counter}")
        

        #++++++++++++++++++++++++++++++++++++++++++

        if savebytrain: 

            pnMo_train = pn + '_Mod_TRAIN.pt'

            best_train = np.nanmin(metrics['Train'][e]) if smallest else np.nanmax(metrics['Train'][e])
            if metrics['Train'][e] == best_train: torch.save(model, pnMo_train)
        
        #******************************************

        if poors is not None: 
            if e >= poors[0] - 1:
                metx = metrics['Validation'][e]
                cuti = metx >= poors[1] if smallest else metx <= poors[1]
                if cuti: 
                    print('POOR MODEL, TERMINATED')
                    counter = patience
                    break
        
        #******************************************

        e += 1
    
    if returnmodel == True: mod = LoadTorch(pnMo)
    
    if pathname == None: os.remove(pnMo), os.remove(pnMe)
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return (metrics, mod) if returnmodel else metrics













def LossWeights(model, tspl, 
                inps, out, out_std, 
                out_bind, metrics_mode, flips, batchsize, per_bin = 20, 
                dtypo = torch.float ): 

    '''
    
    computes per bin loss and determines weights from it

    returns tra_bind shape. 
    
    '''

    tra_bind = out_bind[tspl] 
    groups = tra_bind.shape[1]

    # Precompute unique bins for each group only once
    unique_bins = [np.unique(out_bind[:, g].reshape(-1)) for g in range(groups)]

    samps = []
    samp_dict = {}
    start = 0

    # Vectorized sampling per bin
    for g in range(groups):
        tra_bind_g = tra_bind[:, g].reshape(-1)
        samp_dict[g] = {}
        for u in unique_bins[g]:
            idx = np.where(tra_bind_g == u)[0]
            n = len(idx)
            if n == 0:
                continue
            if n < per_bin:
                choose = tspl[idx]
            else:
                choose = np.random.choice(tspl[idx], size=per_bin, replace=False)
            samps.extend(choose)
            samp_dict[g][u] = np.arange(len(choose)) + start
            start += len(choose)

    samps = np.hstack(samps)

    mini_out, mini_bind = [x[samps] for x in [out, out_bind]]
    mini_std = out_std[samps] if out_std is not None else None
    mini_inp = [x[samps] for x in inps]

    if flips is not None: 
        mini_out = torch.vstack([mini_out, mini_out])
        if mini_std is not None: mini_std = torch.vstack([mini_std, mini_std])
        mini_inp = [torch.vstack([x, torch.flip(x, dims = F)]) for x, F in zip(mini_inp, flips)]
    
    midx = BatchMaker_BW(len(mini_out), batchsize = batchsize, make_full = False) 

    preds = []

    with torch.no_grad():
        model.eval()

        for ib, b, in enumerate(midx):

                mini_inp_b, mini_out_b, mini_std_b, _ = BatchData_BW(b, mini_inp, mini_out, mini_std, out_weights = None)            
                
                mini_inp_b = [xa.to(device, dtype = dtypo) for xa in mini_inp_b]

                pred_b = model(*mini_inp_b).cpu().detach().numpy()

                preds.append(pred_b)
    
    preds = np.concatenate(preds, axis = 0)

    lossweights = np.ones(tra_bind.shape)

    for g in np.arange(groups): 
        
        scores = []
        
        for k, v in samp_dict[g].items(): 
            
            mini_out_k, preds_k = [x[v][:, g] for x in [mini_out, preds]]  
            
            mm_args = copy.deepcopy(metrics_mode[1])
            
            if out_std is not None: 
                mini_std_b = mini_std[v][:, g].numpy()
                mm_args.update({'std': mini_std_b})
            
            score = metrics_mode[0](preds_k, mini_out_k.numpy(), **mm_args)

            scores.append(score)
        
        scores = np.array(scores)           # SCORES ARE POSITIVE AND LOWER IS BETTER

        scores_best = scores.min()

        scores_weights = scores / scores_best
        scores_weights = scores_weights / np.sum(scores_weights)

        # Scale to [1, 10]
        min_w, max_w = scores_weights.min(), scores_weights.max()
        if max_w > min_w:
            scores_weights = 1 + 9 * (scores_weights - min_w) / (max_w - min_w)
        else:
            scores_weights = np.ones_like(scores_weights)

        for k in samp_dict[g].keys(): 

            filt = tra_bind[:, g].reshape(-1) == k

            lossweights[filt, g, :, :] = scores_weights[k]

    
    return lossweights





def LossWeights_Collect(
        used_outs, used_preds, used_std, used_bind, 
        tra_bind,
        metrics_mode, unique_bins): 
    '''
    computes per bin loss and determines weights from it
    returns tra_bind shape. 
    '''
    groups = used_bind.shape[1]
    lossweights = np.ones(tra_bind.shape)

    for g in np.arange(groups): 
        uni_bins_g = unique_bins[g]
        scores = np.zeros(len(uni_bins_g))

        for iu, u in enumerate(uni_bins_g): 
            idx_filt = used_bind[:, g] == u
            if np.sum(idx_filt) > 1: 
                mini_out_k, preds_k = [x[idx_filt][:, g] for x in [used_outs, used_preds]]  
                mm_args = copy.deepcopy(metrics_mode[1])
                if used_std is not None: 
                    mini_std_b = used_std[idx_filt][:, g]
                    mm_args.update({'std': mini_std_b})
                score = metrics_mode[0](preds_k, mini_out_k, **mm_args)
            else: 
                score = 0
            scores[iu] = score

        scores_worst = scores.max()
        scores[scores == 0] = scores_worst
        scores_best  = scores.min()

        scores_weights = scores / scores_best
        scores_weights = scores_weights / np.sum(scores_weights)

        # Scale to [1, 10]
        min_w, max_w = scores_weights.min(), scores_weights.max()
        if max_w > min_w:
            scores_weights = 1 + 9 * (scores_weights - min_w) / (max_w - min_w)
        else:
            scores_weights = np.ones_like(scores_weights)

        for iu, u in enumerate(uni_bins_g): 
            filt = tra_bind[:, g].reshape(-1) == u
            if lossweights.ndim == 2:
                lossweights[filt, g] = scores_weights[iu]
            elif lossweights.ndim == 4:
                lossweights[filt, g, :, :] = scores_weights[iu]
            else:
                raise ValueError("tra_bind (and thus lossweights) must be 2D or 4D.")

    return lossweights


