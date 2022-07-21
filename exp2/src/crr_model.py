#from pytorch_transformers import *
from transformers import *
import torch
import torch.nn as nn
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss
import torch.nn.functional as F

import pdb

class TransformerModel(torch.nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained(args['model_name'])
        #self.bert = AutoModel.from_pretrained(args['model_name'])
        self.num_labels = args['num_labels']
        self.dropout = torch.nn.Dropout(args['dp'])
        self.W = nn.Linear(args['hs'],args['num_labels'])
        self.alpha_param = args['alpha_param']
        self.beta_param = args['beta_param']
        #self.classifier = torch.nn.Linear(args['hs'], args['num_labels'])

            

    
    def forward(self, input_ids, token_type_ids=None,attention_mask=None,labels=None, reps=None,clusters=None):
        _,pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)

        if reps is not None:
            reps.append(pooled_output.detach().cpu())
        # 

        logits = self.W(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self.num_labels),labels)
            #pdb.set_trace()
            if clusters is  None:
                return loss
    
            self.W.weight.data = F.normalize(self.W.weight.data, p=2, dim=1)
            # intra distance (to minimize)
            regTermIntra = 0
            for i in range(len(clusters)):
                cluster_mean = torch.mean(self.W.weight[clusters[i], :], dim=0, keepdim=True)
                ci_dist = torch.cdist(self.W.weight[clusters[i], :],cluster_mean,p=2)
                regTermIntra+= torch.mean(ci_dist)
            # inter distance (to maximize)

            regTermInter = 0
            for i in range(len(clusters)):
                for j in range(i+1,len(clusters)):
                    ci_cj_distance = torch.sum(torch.cdist(
                        self.W.weight[clusters[i],:],self.W.weight[clusters[j],:],p=2))
                    regTermInter += ci_cj_distance/(len(clusters[i])*len(clusters[j]))

            #print('loss:{:.2f}\tregTerm:{:.2f}'.format(loss,regTerm))
            #return (loss + (self.beta_param * regTermIntra)) / (self.alpha_param * regTermInter)
            return loss + (self.beta_param * regTermInter) + (self.alpha_param * regTermIntra)
        else:
            return logits



