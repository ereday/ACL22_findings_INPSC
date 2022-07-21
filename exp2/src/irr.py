import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pdb

import pdb
from tqdm import tqdm, trange
import sys
import numpy as np
import os,random
import math
import time,datetime
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from bert_data import *
from irr_model import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
import itertools

from sklearn.metrics import roc_curve, auc,f1_score,recall_score,precision_score
from sklearn.metrics import precision_recall_fscore_support as prf_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
n_gpu = 1


SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)


def get_datetime_str():
    d = datetime.datetime.now()
    return '%02d_%02d_%02d_%02d-%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute, d.second)

def numpy_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def numpy_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def f1_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = f1_score(golds[:,i],preds[:,i])
    #TODO:: Add overall micro/macro f1 calculation 
    #f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1

def precision_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = precision_score(golds[:,i],preds[:,i])
    #TODO:: Add overall micro/macro f1 calculation 
    #f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1

def recall_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = recall_score(golds[:,i],preds[:,i])
    #TODO:: Add overall micro/macro f1 calculation 
    #f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1


def metrics(all_preds,all_golds,ix2label):
    counts = {ix2label[i]:{'TP':0,'FP':0,'FN':0} for i in range(len(all_golds[0]))}
    for instance_ix in range(len(all_golds)):
        bin_preds = all_preds[instance_ix].tolist()
        bin_golds = [int(x) for x in all_golds[instance_ix].tolist()]

        preds = [ ix2label[pix] for pix,pval in enumerate(bin_preds) if pval ==1]
        golds = [ ix2label[pix] for pix,pval in enumerate(bin_golds) if pval ==1]

        for gold in golds:
            if gold in preds:
                counts[gold]['TP'] +=1
            else:
                counts[gold]['FN'] +=1
                
        for pred in preds:
            if pred not in golds:
                if pred not in counts.keys():
                    counts[pred] = {'TP':0,'FP':0,'FN':0}
                counts[pred]['FP'] +=1
        
    all_tp = sum(counts[k]['TP'] for k in counts.keys())
    all_fp = sum(counts[k]['FP'] for k in counts.keys())
    all_fn = sum(counts[k]['FN'] for k in counts.keys())
    counts['All'] = {'TP':all_tp,'FP':all_fp,'FN':all_fn,'Pre':all_tp/max(0.001,(all_tp+all_fp)),'Rec':all_tp/max(0.001,(all_tp+all_fn))}
    return counts
    
def convert_single_label_format(features,ix2label=None):
    #print("convert_single_label_format inside")
    #pdb.set_trace()
    label_result = []
    for f in features:
        for ix,val in enumerate(f.label_ids):
            if val == 1.0:
                label_result.append(ix)
    return torch.tensor(label_result,dtype=torch.long)


def main_loop(model,optimizer,schedular,train_dataloader,eval_dataloader,args,logger,ix2label,clusters):
    global_step = 0
    model.train()
    best_micro_auc = -1
    keep_best_results = {}
    min_epoch_num = int(args['min_num_train_epochs'])
    missed_epoch_num = 0
    label2clusterIndex = {l:(cix+1) for cix,cluster in enumerate(clusters) for l in cluster}
    cluster_num = max(label2clusterIndex.values())
    for i_ in tqdm(range(int(args['max_num_train_epochs'])), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            ll = label_ids.tolist()
            batch_clusters = [[] for i in range(cluster_num)]
            for (lix,l) in enumerate(ll):
                batch_clusters[label2clusterIndex[l]-1].append(lix)
            loss = model(input_ids, segment_ids, input_mask, label_ids,clusters=clusters,batch_clusters=batch_clusters)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            schedular.step()
            optimizer.zero_grad()
            global_step += 1
        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        result = evaluate(model,eval_dataloader,args,logger,ix2label)
        if result['f1']['All']['Rec'] > best_micro_auc:
            keep_best_results = result
            best_micro_auc = result['f1']['All']['Rec']
            print("saved model at {} epoch with {} recall value".format(i_+1,best_micro_auc))
            output_eval_file = args['output_dir']
            with open(output_eval_file, "a") as writer:
                writer.write('\n--------------------------------------------\n')
                writer.write("saved model(dev set): {:.3f}\t{:.3f}\t{:.3f}".format(result['f1']['All']['Pre'],result['f1']['All']['Rec'],result['f1']['All']['F1']))
                writer.write('\n--------------------------------------------\n')

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), args["output_model_file"])
            missed_epoch_num = 0
        else:
            missed_epoch_num +=1
            if missed_epoch_num > args['consecutive_miss'] and i_ >= min_epoch_num:
                # stop training
                break
            # print info and model saved message
    # Save final model
    model_to_save = model.module if hasattr(model, 'module') else model  
    torch.save(model_to_save.state_dict(), args["output_model_file"]+'.final')
    return result,keep_best_results

def tsne_visualize(reps,labels,preds,fname='figure.png'):
    target_label = labels
    label_count = len(target_label[0])
    target_label = np.array([[ix for ix,v in enumerate(lab) if v ==1.0] for lab in target_label])
    feat_cols = ['feat' + str(i) for i in range(reps.shape[1])]
    df = pd.DataFrame(reps, columns=feat_cols)
    df['y'] = target_label
    df['label'] = df['y'].apply(lambda i: str(i))

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(df[feat_cols].values)

    #pca = PCA(n_components=3)
    #pca_result = pca.fit_transform(df[feat_cols].values)
    #df['pca-one'] = pca_result[:, 0]
    #df['pca-two'] = pca_result[:, 1]
    #df['pca-three'] = pca_result[:, 2]
    #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    #plt.figure(figsize=(16, 10))
    #sns.scatterplot(
    #    x="pca-one", y="pca-two",
    #    hue="y",
    #    palette=sns.color_palette("hls", 7),
    #    data=df,
    #    legend="full",
    #    alpha=0.3
    #)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #tsne_results = tsne.fit_transform(df[feat_cols].values)
    tsne_results = tsne.fit_transform(pca_result_50)
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    # sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", label_count),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig(fname)

def evaluate(model,eval_dataloader,args,logger,ix2label,is_final=False):
    all_logits = None
    all_labels = None
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    representations = [] if is_final else None
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask, reps=representations)

        converted_label_ids = []
        for i in range(0, len(label_ids)):
            ix = label_ids[i]
            tmp = [0.0]*logits.shape[1]
            tmp[ix] = 1.0
            converted_label_ids.append(tmp)
        converted_label_ids = torch.tensor(converted_label_ids).cuda()


        #tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        tmp_eval_accuracy = accuracy_thresh_softmax(logits, converted_label_ids,use_softmax=True)

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = converted_label_ids.detach().cpu().numpy()
            #all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, converted_label_ids.detach().cpu().numpy()), axis=0)
            #all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        #print("eval example size:",input_ids.size(0))
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples


    #print(all_logits)
    #print(all_logits.shape)
    #f1_preds= 1 * (numpy_sigmoid(all_logits) > 0.5)


    #f1_preds = 1 * (softmax(torch.tensor(all_logits), dim=1).numpy() > 0.5)
    #f1_preds = np.zeros_like(all_logits,dtype=int)
    #f1_preds[np.arange(f1_preds.shape[0]), ones_indices] = 1

    pred_indices = np.argmax(softmax(torch.tensor(all_logits), dim=1).numpy(),axis=1)
    gold_indices = np.argmax(all_labels,axis=1)
    precision,recall,fscore,_ = prf_score(gold_indices,pred_indices)
    m_precision,m_recall,m_f1,_ = prf_score(gold_indices,pred_indices,average='macro')
    f1 = {}
    for ix in range(len(recall)):
        f1[ix2label[ix]] = {'Pre':float("{:.3f}".format(precision[ix])),
                            'Rec':float("{:.3f}".format(recall[ix])),
                            'F1':float("{:.3f}".format(fscore[ix]))}

    f1['All'] = {'Pre':float("{:.3f}".format(m_precision)),
                 'Rec':float("{:.3f}".format(m_recall)),
                 'F1':float("{:.3f}".format(m_f1))}
    #pdb.set_trace()
    #f1 = metrics(f1_preds, all_labels, ix2label)
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              #               'loss': tr_loss/nb_tr_steps,
              'f1': f1, }
    # 'roc_micro':roc_auc['micro']}

    output_eval_file = args['output_dir']
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        if is_final == True:
            writer.write("\n ----- Best model on test set ----- \n")
            writer.write('\nparams:\n')
            writer.write(str(args))
            writer.write('\n')
        for key in sorted(result.keys()):
            if key != 'f1':
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            else:
                counts = result[key]
                for k in counts.keys():
                    logger.info(" %s = %s", k, counts[k])
                    writer.write("%s = %s\n" % (k, str(counts[k])))
                print("###################")

        writer.write("\n ----------------------------------------------------\n")

    if is_final:
        reps = torch.cat(representations,dim=0)
        return result,all_logits,all_labels,ix2label,reps
    else:
        return result

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def accuracy_thresh_softmax(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, use_softmax:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if use_softmax: y_pred = softmax(y_pred)
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()


def convert_label_for_cv(df):
    labelconv = {}
    converted_labels = []
    for ix,row in df.iterrows():
        t = row[1:].tolist()
        t2 = [str(i) for i in t]
        orgl = "".join(t2)
        if orgl not in labelconv:
            labelconv[orgl] = len(labelconv)
        converted_labels.append(labelconv[orgl])
    df['converted_label'] = converted_labels
    return df


def run_fold_experiment(params,trn_df,val_df,tst_df,is_final=False):

    args = {
        "train_size": params['bs'],
        "val_size": params['bs'],
        'model_name': params['model_name'],
        "do_lower_case":False,                
        "max_seq_length": params['max_seq_len'],
        "do_train": True,
        "do_eval": True,
        "train_batch_size": params['bs'],
        "eval_batch_size": params['bs'],
        "learning_rate": params['lr'],            
        "max_num_train_epochs": params['max_epoch'],
        "min_num_train_epochs": params['min_epoch'],
        'consecutive_miss': params['consecutive_miss'],
        "warmup_proportion": params['warmup_proportion'],
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "hs":params['hs'],
        "dp":params['dp'],
        'output_dir':params['output_dir'],
        'output_model_file':params['output_model_file'],
        'only_generate':params['only_generate'],
        'alpha_param':params['alpha_param'],
        'beta_param': params['beta_param'],
    }

    processors = {
        "hcs_multilabel": HcsMultiLabelTextProcessor
    }

    task_name= "hcs_multilabel"        
    processor = processors[task_name](trn_df,val_df)
    label_list = processor.labels
    num_labels = len(label_list)
    args["label_list"] = label_list
    args["num_labels"] = num_labels

    tokenizer = BertTokenizer.from_pretrained(args['model_name'],do_lower_case=args['do_lower_case'])
    #tokenizer = AutoTokenizer.from_pretrained(args['model_name'], do_lower_case=False, use_fast=False, pad_token='[PAD]')
    train_examples = processor.get_train_examples(trn_df, size=args['train_size'])
    train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)        
    num_train_steps = int(len(train_examples) / args['train_batch_size']) * args['max_num_train_epochs']
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args['train_batch_size'])
    logger.info("  Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    # len(train_features[0].label_ids) = 62
    #all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
    all_label_ids = convert_single_label_format(train_features,ix2label=processor.ix2label)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

    labels = sorted(list(processor.ix2label.values()))
    clusters = []
    major_labels = sorted(list(set([(label - (label%100)) for label in labels])))
    for mj in major_labels:
        cluster = []
        for ix,label in enumerate(labels):
            if abs(mj-label) < 50:
                cluster.append(ix)
        clusters.append(cluster)
    # development data
    eval_examples = processor.get_dev_examples(val_df, size=args['val_size'])
    eval_features = convert_examples_to_features(eval_examples, label_list, args['max_seq_length'], tokenizer)    
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    all_label_ids = convert_single_label_format(eval_features,ix2label=processor.ix2label)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # test data
    test_examples = processor.get_test_examples(tst_df, size=args['val_size'])
    test_features = convert_examples_to_features(test_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    #all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.float)
    all_label_ids = convert_single_label_format(test_features,ix2label=processor.ix2label)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

    # CREATE MODEL 
    model = TransformerModel(args)    
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['learning_rate'],
                      correct_bias = True)
    schedular = WarmupLinearSchedule(optimizer,warmup_steps=args['warmup_proportion'],t_total=t_total)
    


    # Main Loop
    if  args['only_generate']:
        final_model_report, best_model_report = None,None
    else:
        final_model_report,best_model_report = main_loop(model,optimizer,schedular,train_dataloader,eval_dataloader,args,logger,processor.ix2label,clusters)

    # Foo deneme # final icin
    if is_final:
        model_state_dict = torch.load(args["output_model_file"])
        #model_state_dict = torch.load(args["output_model_file"]+'.final')
        model = TransformerModel(args)
        model.load_state_dict(model_state_dict)
        model.to(device)
        result,all_logits,all_labels,ix2label,representations = evaluate(model,test_dataloader,args,logger,processor.ix2label,is_final=True)

        #all_preds = 1 * (softmax(torch.tensor(all_logits), dim=1).numpy() > 0.5)
        all_preds = np.zeros_like(all_logits, dtype=int)
        ones_indices = np.argmax(softmax(torch.tensor(all_logits), dim=1).numpy(), axis=1)
        all_preds[np.arange(all_preds.shape[0]), ones_indices] = 1

        reps = representations.numpy()
        figname=params['probability_output_file']+'.png'
        #tsne_visualize(reps,all_labels,all_preds,fname=figname)
        #all_probs = pd.DataFrame(numpy_sigmoid(all_logits),columns=val_df.columns[1:])
        all_probs = pd.DataFrame(softmax(torch.tensor(all_logits),dim=1).numpy(), columns=val_df.columns[1:])
        all_probs['Input'] = val_df['input']
        csv_out_file_name = params['exp_date']+'_'+params['probability_output_file']
        csv_out_dir = params['output_dir'].split(params['exp_date'])[0]
        all_probs.to_csv(os.path.join(csv_out_dir,csv_out_file_name),sep=',',index=None)

    
    return final_model_report,best_model_report


def train_final_model(params):
    trn_df_ = pd.read_table(os.path.join(params['data_dir'],'train.csv'),sep=',')
    val_df = pd.read_table(os.path.join(params['data_dir'],'dev.csv'),sep=',')
    tst_df = pd.read_table(os.path.join(params['data_dir'],'test.csv'),sep=',')
    # Ignore major classes
    labels_to_drop = [i for i in tst_df.columns.tolist()[1:] if int(i) % 100 == 0]
    trn_df_ = trn_df_.drop(labels_to_drop,axis=1)
    val_df = val_df.drop(labels_to_drop, axis=1)
    tst_df = tst_df.drop(labels_to_drop, axis=1)
    label_num = len(trn_df_.columns)-1

    exp_date = get_datetime_str()
    base_output_dir = params['output_dir']
    base_model_file = params['output_model_file']
    params['exp_date'] = exp_date
    trn_df  = convert_label_for_cv(trn_df_)
    

    split_ix = 1
    log_final_models = {'f1':{}}
    log_best_models = {'f1':{}}


    if params['only_generate']:
        final_model_report, best_model_report = run_fold_experiment(params, trn_df, val_df, tst_df,is_final=True)
        return None, None
    out_link = params['probability_output_file'].split('.csv')[0].replace(' ', '_')
    log_file_name = '{}_{}_final.log'.format(exp_date,out_link)
    model_f_name = '{}_{}_final.bin'.format(exp_date,out_link)
        
    params['output_dir'] = os.path.join(base_output_dir,log_file_name)
    params['output_model_file'] = os.path.join(base_model_file,model_f_name)

    #"output_dir": './bert_saved_models/{}_report.log'.format(exp_date),

    #print("before run_fold_experiment")
    #pdb.set_trace()
    final_model_report,best_model_report = run_fold_experiment(params,trn_df,val_df,tst_df,is_final=True)

        
        
    for k in final_model_report['f1'].keys():            
        if k not in log_final_models['f1']:
            log_final_models['f1'][k] = final_model_report['f1'][k]
        else:
            for k2 in final_model_report['f1'][k]:
                log_final_models['f1'][k][k2] += final_model_report['f1'][k][k2]
    for k in best_model_report['f1'].keys():
        if k not in log_best_models['f1']:
            log_best_models['f1'][k] = best_model_report['f1'][k]
        else:
            for k2 in best_model_report['f1'][k]:
                log_best_models['f1'][k][k2]  += best_model_report['f1'][k][k2]

        
    for k in log_final_models['f1'].keys():
        for k2 in log_final_models['f1'][k].keys():
            log_final_models['f1'][k][k2] = float("{:.2f}".format(log_final_models['f1'][k][k2]))
            #log_best_models['f1'][k][k2] = float("{:.2f}".format(log_best_models['f1'][k][k2]))

    for k in log_best_models['f1'].keys():
        for k2 in log_best_models['f1'][k].keys():
            log_best_models['f1'][k][k2] = float("{:.2f}".format(log_best_models['f1'][k][k2]))

    return log_final_models,log_best_models

def main_final():
    if len(sys.argv) <3:
        sys.exit(1)
    
    experiments = {'params':[],'reports':[]}
    
    output_file = sys.argv[2]
    params = { 'output_dir':'./outputs/irr_models',
               'output_model_file':'./outputs/irr_models',
               'only_generate': False,
               'probability_output_file':output_file,
               'bs':16,
                'lr': float(sys.argv[7]),
               'min_epoch':15,
               'max_epoch': 30,
               'consecutive_miss':3,
               'warmup_proportion':0.1,
               'hs':768,
                'dp':float(sys.argv[8]),
               'alpha_param':float(sys.argv[5]),
               'beta_param': float(sys.argv[6]),
               'max_seq_len':200,
               'model_name': sys.argv[3], 
               'data_dir': '../data/{}/'.format(sys.argv[4]),
    }
    print(params)
    report = train_final_model(params)
    print(report)
    print("-----------------")
    print(params)
    print(output_file)



if __name__ == '__main__':
    main_final()
    


