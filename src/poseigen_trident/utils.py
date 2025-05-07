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
    



###################################################################################################

#### TRAINER ####

def trident_loss_mode_default(): return [se.AError, {'expo': 2}]

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
    out_std = None, out_weights = None, out_bind = None, 
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
    batchsize = 128, opt = Adam, learningrate = 0.001, maxepochs = 20, patience = 5, 
    pathname = None, statusprints = True, returnmodel = False, pickup = False,
    savebytrain = False
    ): 
    
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
    t_ns, v_ns = (np.append(np.arange(0, y, batchsize),y) for y in [tlength,vlength])

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
            #if out_bind_b is not None and loss_bind: lm_args.update({'bind': out_bind_b.to(device, dtype = dtypo)})
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
                #if metbatches[0][4] is not None: mm_args.update({'bind': np.vstack([mb[4].numpy() for mb in metbatches])})
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

        for keyo in ['out_std', 'out_weights', 'out_bind']: 
            if keyo not in trainer_args.keys(): trainer_args[keyo] = None

        inps, out, out_std, out_weights, out_bind = [trainer_args[x] 
                                                     for x in ['inps', 'out', 
                                                               'out_std', 'out_weights', 'out_bind']]
                        
        if 'batchsize' not in list(add_pred_args.keys()): 
            if 'batchsize' in aak: add_pred_args['batchsize'] = trainer_args['batchsize']
            else: add_pred_args['batchsize'] = 128 # DEFAULT

        if 'flips' not in list(add_pred_args.keys()): 
            if 'flips' in trainer_args: add_pred_args['flips'] = trainer_args['flips']

        add_pred_args['rewrite'] = pred_rewrite # TAKES PRECENDENCE


        ####################################################################################################
        
        idx_sel = trainer_args['Split'][score_on]
        out_sel = out[idx_sel]
        
        ext, inps_predo = '', inps
        if score_only: 
            ext = '_only_' + str(score_on)
            inps_predo = [qer[idx_sel] for qer in inps]

        preds = TridentPredMulti(pathname + '_Mod.pt', inps_predo, 
                                 pn_preds = pathname + '_Preds' + ext + '.p', 
                                 **add_pred_args)

        preds_sel = preds if score_only else preds[idx_sel]

        mm_args = copy.deepcopy(metrics_mode[1])
        if out_std is not None: mm_args.update({'std': out_std[idx_sel]})
        if out_bind is not None: mm_args.update({'bind': out_bind[idx_sel]})

        met = metrics_mode[0](preds_sel, out_sel, **mm_args)

        ###################################################################################################
    
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
                       score_on = 2, score_only = False,

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
                'score_on': score_on, 'score_only': score_only,
                'add_pred_args': add_pred_args}
    
    for c in modelcombos: 
        s, r = c
        
        if newmetrics[s][r] is None:
           
            if statusprints == True: print(f' Cross val {s+1} of {lsp}, Repeat {r+1} of {repeats}')

            if newpathname is not None and savemodels == True: savepath = newpathname + str(s) + '_' + str(r)

            newmetrics[s][r] = TridentCanScorer(algo, algo_args, data, Split = Splits[s],
                                                **TCS_args, pathname = savepath)

            if statusprints == True: print(f"{newmetrics[s][r]}")

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

    if pyt: inps = torch.split(inp, lenba, byaxis)
    else: inps = np.split(inp, lenba, byaxis)

    if mode is not None: inps = [mode[0](np, **mode[1]) for np in inps]

    return inps


def BinnedLoss(inp1, inp2, std = None, 
                weights = None, useweights = True, 

                bind = None, 
                byaxis = None, seperate = False,

                metrics_mode = [se.AError, {'expo': 2}],
                summarize_mode = [mex.MeanExpo, {'expo': 2}], 
                pyt = False):
    
    #uni is determined by the input always. 
    
    weio = True if weights is not None and isinstance(weights, int) is False and useweights else False

    inp = [inp1, inp2] if std is None else [inp1, inp2, std]
    if weio: inp = [*inp, weights]

    if byaxis is not None:
        inp_ba = [ByAxis(p, byaxis = byaxis, pyt = pyt) 
                  for p in inp] #so now a list of lists, each list having seperated by axis 
        inp_pairs = [[p[i] for p in inp_ba] for i in range(len(inp_ba[0]))]
        bind = ByAxis(bind, byaxis = byaxis, pyt = False) #BY DEFAULT IT IS NUMPY
    else: 
        inp_pairs = [inp]
        bind = [bind]
    
    uni =  [np.unique(b) for b in bind]

    gxs = []
    
    for ip, bi, un in zip(inp_pairs, bind, uni):
    
        ip2 = ip #############[b.reshape(-1) for b in ip] #########################
        bi2 = bi.reshape(-1)

        par = bm.BinParser(bi2, uni = un, categorical = True)

        if weio:
            gx = [metrics_mode[0](*[b[a] for b in ip2[:-1]], weights = ip2[-1][a], 
                                  **metrics_mode[1])
                                  for a in par if len(a) > 1]
        else: 
            gx = [metrics_mode[0](*[b[a] for b in ip2], 
                                  **metrics_mode[1])
                                  for a in par if len(a) > 1]
        
        gx = torch.stack(gx) if pyt else np.array(gx)
        
        gxs.append(gx)
    
    if summarize_mode is not None: 
    
        if seperate: 
            gx_sums = [summarize_mode[0](gx, **summarize_mode[1]) for gx in gxs]
        else: 
            gxs = torch.stack(gxs) if pyt else np.stack(gxs)
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
    
    return np.vstack(alls)




def TridentWindow(inp, size): 
    ST = np.lib.stride_tricks.sliding_window_view
    winshape = (inp.shape[1], size, inp.shape[-1])
    sig_win = ST(inp, (1, *winshape))
    sig_win_rs = sig_win.reshape(-1, *winshape)
    return sig_win_rs




