from pyscipopt import Model, quicksum
import random as random
import math
import pandas as pd
import pdb
import numpy as np
import sys
from metrics import metrics
            
def read_probs_2_list(fname):
    l2ix = {}
    df = pd.read_table(fname,sep=',')
    labels = df.columns[:-1].tolist()    
    for _label in labels:
        label = int(_label)
        if label % 100 == 0:
            # major class
            #l2ix[label] = {}
            l2ix[_label] = []
        else:
            maj_label = label - (label % 100)
            l2ix[str(maj_label)].append(_label)
            #l2ix[maj_label][label] = len(l2ix[maj_label])+1
    return l2ix,df


def run_ILP(p,p1,l2ix):
    #create model
    model = Model("model")

    #decision variables
    A = {} # if major class is selected or not
    b = {} # if minor class is selected or not

    Major_Classes = l2ix.keys()
    for x in Major_Classes:
        A[x] = model.addVar("A(%s)"%(x),vtype="BINARY")
        #for y in range(1,minor_class_num[x]+1):
        for y in l2ix[x]:
            b[x,y] = model.addVar("b(%s,%s)" % (x,y), vtype="BINARY")
        
    for x in Major_Classes:
        # If Ax==1,then sum(axy)>=1
        model.addCons(quicksum(b[x,y] for y in l2ix[x]) >= A[x])

        # if axy==1, then Ax=1
        for y in l2ix[x]:
            model.addCons(b[x,y] <= A[x]) 

    # Objective function
    res = []
    for x in Major_Classes:
        res.append(A[x]*math.log(max(1e-20,p1[x])) +(1-A[x])* math.log(max(1e-20,1-p1[x])))
        for y in l2ix[x]:
            res.append(b[x,y]*math.log(max(1e-20,p[x,y])) +(1-b[x,y])* math.log(max(1e-20,1-p[x,y])))

    # Maximize objective function
    model.setObjective(quicksum(res),"maximize")        

    model.optimize()
    model.getObjVal()
    status= model.getStatus()
    print(status)
    print("Objective Value = %s" %model.getObjVal())


    result = {}
    for x in Major_Classes:
        result[x] = model.getVal(A[x])
        for y in l2ix[x]:
            result[y] = model.getVal(b[x,y])
    return result

    
def main(fname,gold_file):
    l2ix,df = read_probs_2_list(fname)
    Major_Classes = l2ix.keys()
    categories = df.columns[:-1].tolist()
    result_all = []
    for ix,row in df.iterrows():
        # Parameters
        p  = {}
        p1 = {}
        for x in Major_Classes:
            p1[x] = row[x]
            for y in l2ix[x]: 
                p[x,y] = row[y] 
        result_row = run_ILP(p,p1,l2ix)
        result_all.append([int(result_row[cat]) for cat in categories])

    ILP_results = pd.DataFrame(result_all,columns=categories)
    ILP_results['Input'] = df['Input']
    parts = fname.split('.csv')
    outname = parts[0]+'_ILP.csv'
    ILP_results.to_csv(outname,index=False,sep=',')


if __name__ == '__main__':
    print("Reminder: Do not forget to set SCIPOPTDIR and DYLD_LIBRARY_PATH variables")
    if len(sys.argv) <2:
        print("Usage: python ILP_decoder.py [PROBABILITY_FILE]")
        print("e.g.: python ILP_decoder.py bilstmattention_hle_test_output.csv or python ILP_decoder.py bert_plain_test_output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file,'test.csv')
    
#for model_name in ['lstm','bilstm','bilstmattention','transformer']:
#    for type_ in ['plain','hle']:
#        main('{}_{}_test_output.csv'.format(model_name,type_),'test.csv')

