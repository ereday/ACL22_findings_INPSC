# from pytorch_transformers import *
from transformers import *
import torch
import torch.nn as nn
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
import pdb


class TransformerModel(torch.nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained(args['model_name'])
        self.num_labels = args['num_labels']
        self.dropout = torch.nn.Dropout(args['dp'])

        self.use_knowledge = args['use_knowledge']
        if self.use_knowledge:
            self.V = nn.Linear(args['num_labels'], args['hs'])
        else:
            self.W = nn.Linear(args['hs'], args['num_labels'])

        # self.classifier = torch.nn.Linear(args['hs'], args['num_labels'])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, Smatrix=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)

        #
        if Smatrix is not None:
            W_t = self.V(Smatrix)
            W = W_t.transpose(0, 1)
            logits = torch.matmul(pooled_output, W)
        else:
            logits = self.W(pooled_output)

        # logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            # pdb.set_trace()
            return loss
        else:
            return logits

