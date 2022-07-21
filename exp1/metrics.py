import numpy as np
from sklearn.metrics import roc_curve, auc,f1_score,recall_score,precision_score
import pdb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
def f1_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = f1_score(golds[:,i],preds[:,i])
    return f1

def precision_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = precision_score(golds[:,i],preds[:,i])
    return f1

def recall_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = recall_score(golds[:,i],preds[:,i])
    return f1

def metrics(preds,golds):
    res = np.equal(preds,golds)
    acc = np.mean(res,axis=1).sum()/golds.shape[0]
    weights = sum(golds).tolist()
    f1  = f1_measures(preds,golds)
    ttl = 0
    for i in range(len(weights)):
        ttl += (f1[i]*weights[i])
    ttl = ttl/sum(weights)
    f1['weighted'] = ttl

    p1 = precision_measures(preds,golds)
    ttl = 0
    for i in range(len(weights)):
        ttl += (p1[i]*weights[i])
    ttl = ttl/sum(weights)
    p1['weighted'] = ttl    

    r1 = recall_measures(preds,golds)
    ttl = 0
    for i in range(len(weights)):
        ttl += (r1[i]*weights[i])
    ttl = ttl/sum(weights)
    r1['weighted'] = ttl
    
    for u in f1.keys():
        f1[u] = float(" {:.3f}".format(f1[u]))
        p1[u] = float(" {:.3f}".format(p1[u]))
        r1[u] = float(" {:.3f}".format(r1[u]))
    result = {'accuracy': float("{:.3f}".format(acc)),
              'f1': f1,
              'precision':p1,
              'recall':r1}
    return result



def metrics_partial(preds,golds,ix2lab,label_array=[]):
    res = np.equal(preds,golds)
    acc = np.mean(res,axis=1).sum()/golds.shape[0]
    weights = sum(golds).tolist()
    f1  = f1_measures(preds,golds)
    ttl = 0
    sumweight = 0 
    for i in range(len(weights)):
        if ix2lab[i] not in label_array:
            continue
        ttl += (f1[i]*weights[i])
        sumweight += weights[i]
    ttl = ttl/sumweight
    #ttl = ttl/sum(weights)
    f1['weighted'] = ttl

    p1 = precision_measures(preds,golds)
    ttl = 0
    sumweight =0 
    for i in range(len(weights)):
        if ix2lab[i] not in label_array:
            continue
        ttl += (p1[i]*weights[i])
        sumweight += weights[i]
    ttl = ttl/sumweight
    #ttl = ttl/sum(weights)
    p1['weighted'] = ttl    

    r1 = recall_measures(preds,golds)
    ttl = 0
    sumweight = 0
    for i in range(len(weights)):
        if ix2lab[i] not in label_array:
            continue
        ttl += (r1[i]*weights[i])
        sumweight += weights[i]
    ttl = ttl/sumweight 
    #ttl = ttl/sum(weights)
    r1['weighted'] = ttl

    f1_partial,p1_partial,r1_partial = {},{},{}
    for u in f1.keys():
        if u !='weighted' and ix2lab[u] not in label_array:
            continue
        f1_partial[u] = float(" {:.3f}".format(f1[u]))
        p1_partial[u] = float(" {:.3f}".format(p1[u]))
        r1_partial[u] = float(" {:.3f}".format(r1[u]))
    result = {'accuracy': float("{:.3f}".format(acc)),
              'f1': f1_partial,
              'precision':p1_partial,
              'recall':r1_partial}
    return result
