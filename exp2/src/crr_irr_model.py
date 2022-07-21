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
        self.theta_param = args['theta_param']
        #self.classifier = torch.nn.Linear(args['hs'], args['num_labels'])

            

    
    def forward(self, input_ids, token_type_ids=None,attention_mask=None,labels=None, reps=None,clusters=None,batch_clusters=None):
        _,pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)

        if reps is not None:
            reps.append(pooled_output.detach().cpu())
        # pooled_output.shape => bsxhs = 16x768

        logits = self.W(pooled_output)

        if labels is None:
            return logits
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self.num_labels),labels)

            if clusters is  None:
                return loss

            self.W.weight.data = F.normalize(self.W.weight.data, p=2, dim=1)
            ## intra distance (to minimize)
            regTermIntra = 0
            for i in range(len(clusters)):
                cluster_mean = torch.mean(self.W.weight[clusters[i], :], dim=0, keepdim=True)
                ci_dist = torch.cdist(self.W.weight[clusters[i], :],cluster_mean,p=2)
                regTermIntra+= torch.mean(ci_dist)
            ## inter distance (to maximize)
            regTermInter = 0
            for i in range(len(clusters)):
                for j in range(i+1,len(clusters)):
                    ci_cj_distance = torch.sum(torch.cdist(
                        self.W.weight[clusters[i],:],self.W.weight[clusters[j],:],p=2))
                    regTermInter += ci_cj_distance/(len(clusters[i])*len(clusters[j]))#

            # Bert representation constrain
            if batch_clusters is not None:
                cosine_values = F.cosine_similarity(pooled_output.unsqueeze(1), pooled_output.unsqueeze(0), dim=2)
                extra_loss = 0
                cntr = 0
                for i in range(len(batch_clusters)):
                    for j in range(len(batch_clusters)):
                        if i == j:
                            continue
                        c1 = batch_clusters[i]
                        c2 = batch_clusters[j]
                        for k in range(len(c1)):
                            for l in range(len(c1)):
                                if k == l:
                                    continue

                                for m in range(len(c2)):
                                    extra_loss += max(0, (cosine_values[c1[k]][c2[m]] - cosine_values[c1[k]][c1[l]]))
                                    cntr += 1
                                    # print("({},{}) - ({},{})".format(c1[k],c2[m],c1[k],c1[l]))
            #return ((loss + (self.beta_param * regTermIntra)) / (self.alpha_param * regTermInter) ) + self.theta_param*(extra_loss / cntr)
            return loss + (self.beta_param * regTermInter) + (self.alpha_param * regTermIntra) + self.theta_param*(extra_loss / max(cntr,0.1))
            #return loss + extra_loss/max(cntr,1)
