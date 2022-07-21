import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pdb

import pdb
from tqdm import tqdm, trange
import sys
import numpy as np
import os, random
import math
import time, datetime

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from bert_data import *
from bert_model_wo_reg import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import itertools

from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 1

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)


def get_datetime_str():
    d = datetime.datetime.now()
    return '%02d_%02d_%02d_%02d-%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def numpy_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f1_measures(preds, golds):
    f1 = dict()
    for i in range(golds.shape[1]):
        f1[i] = f1_score(golds[:, i], preds[:, i])
    # TODO:: Add overall micro/macro f1 calculation
    # f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1


def precision_measures(preds, golds):
    f1 = dict()
    for i in range(golds.shape[1]):
        f1[i] = precision_score(golds[:, i], preds[:, i])
    # TODO:: Add overall micro/macro f1 calculation
    # f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1


def recall_measures(preds, golds):
    f1 = dict()
    for i in range(golds.shape[1]):
        f1[i] = recall_score(golds[:, i], preds[:, i])
    # TODO:: Add overall micro/macro f1 calculation
    # f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1


def metrics(all_preds, all_golds, ix2label):
    counts = {ix2label[i]: {'TP': 0, 'FP': 0, 'FN': 0} for i in range(len(all_golds[0]))}
    for instance_ix in range(len(all_golds)):
        bin_preds = all_preds[instance_ix].tolist()
        bin_golds = [int(x) for x in all_golds[instance_ix].tolist()]

        preds = [ix2label[pix] for pix, pval in enumerate(bin_preds) if pval == 1]
        golds = [ix2label[pix] for pix, pval in enumerate(bin_golds) if pval == 1]

        for gold in golds:
            if gold in preds:
                counts[gold]['TP'] += 1
            else:
                counts[gold]['FN'] += 1

        for pred in preds:
            if pred not in golds:
                if pred not in counts.keys():
                    counts[pred] = {'TP': 0, 'FP': 0, 'FN': 0}
                counts[pred]['FP'] += 1

    all_tp = sum(counts[k]['TP'] for k in counts.keys())
    all_fp = sum(counts[k]['FP'] for k in counts.keys())
    all_fn = sum(counts[k]['FN'] for k in counts.keys())
    counts['All'] = {'TP': all_tp, 'FP': all_fp, 'FN': all_fn, 'Pre': all_tp / max(0.001, (all_tp + all_fp)),
                     'Rec': all_tp / max(0.001, (all_tp + all_fn))}
    return counts


def main_loop(model, optimizer, schedular, train_dataloader, eval_dataloader, args, logger, ix2label, Smatrix,
              num_epocs=1):
    global_step = 0
    model.train()
    best_micro_auc = -1
    keep_best_results = {}
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, Smatrix=Smatrix)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            schedular.step()
            optimizer.zero_grad()
            global_step += 1
        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_ + 1))
        result = evaluate(model, eval_dataloader, args, logger, ix2label, Smatrix)
        if result['f1']['All']['Rec'] > best_micro_auc:
            keep_best_results = result
            best_micro_auc = result['f1']['All']['Rec']
            print("saved model at {} epoch with {} micro-auc value".format(i_ + 1, best_micro_auc))
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), args["output_model_file"])
            # print info and model saved message
    # Save final model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), args["output_model_file"] + '.final')
    return result, keep_best_results


def evaluate(model, eval_dataloader, args, logger, ix2label, Smatrix, is_final=False):
    all_logits = None
    all_labels = None
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, Smatrix=Smatrix)
            logits = model(input_ids, segment_ids, input_mask, Smatrix=Smatrix)

        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        # print("eval example size:",input_ids.size(0))
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    #     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(args["num_labels"]):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # print(all_logits)
    # print(all_logits.shape)
    f1_preds = 1 * (numpy_sigmoid(all_logits) > 0.5)
    f1 = metrics(f1_preds, all_labels, ix2label)
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
        return result, all_logits, all_labels, ix2label
    else:
        return result


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
          sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()


def get_predictions_from_out_file(fname_preds, fname_golds, thresh=0.5):
    df = pd.read_csv(fname_preds)
    result_mx = df.loc[:, '100':'900'].as_matrix()
    preds = 1 * (result_mx > thresh)
    df_gold = pd.read_table(fname_golds)
    result_mx_gold_ = [x.split(' ') for x in df_gold.loc[:, 'major_classes'].tolist()]
    result_mx_gold = np.array(result_mx_gold_)
    a1 = result_mx_gold.astype(int)
    a2 = preds.astype(int)
    res = np.equal(a1, a2)


def convert_label_for_cv(df):
    labelconv = {}
    converted_labels = []
    for ix, row in df.iterrows():
        t = row[1:].tolist()
        t2 = [str(i) for i in t]
        orgl = "".join(t2)
        if orgl not in labelconv:
            labelconv[orgl] = len(labelconv)
        converted_labels.append(labelconv[orgl])
    df['converted_label'] = converted_labels
    return df


def run_fold_experiment(params, trn_df, val_df, Smatrix, is_final=False):
    args = {
        "train_size": params['bs'],
        "val_size": params['bs'],
        'model_name': 'bert-base-german-cased',
        "do_lower_case": False,
        "max_seq_length": params['max_seq_len'],
        "do_train": True,
        "do_eval": True,
        "train_batch_size": params['bs'],
        "eval_batch_size": params['bs'],
        "learning_rate": params['lr'],
        "num_train_epochs": params['epoch'],
        "warmup_proportion": params['warmup_proportion'],
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "hs": params['hs'],
        "dp": params['dp'],
        'output_dir': params['output_dir'],
        'output_model_file': params['output_model_file'],
        'use_knowledge': params['use_knowledge']
    }

    processors = {
        "hcs_multilabel": HcsMultiLabelTextProcessor
    }

    task_name = "hcs_multilabel"
    processor = processors[task_name](trn_df, val_df)
    label_list = processor.labels
    num_labels = len(label_list)
    args["label_list"] = label_list
    args["num_labels"] = num_labels
    tokenizer = BertTokenizer.from_pretrained(args['model_name'], do_lower_case=args['do_lower_case'])
    train_examples = processor.get_train_examples(trn_df, size=args['train_size'])
    train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)
    num_train_steps = int(len(train_examples) / args['train_batch_size']) * args['num_train_epochs']
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args['train_batch_size'])
    logger.info("  Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

    # development data
    eval_examples = processor.get_dev_examples(val_df, size=args['val_size'])
    eval_features = convert_examples_to_features(eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

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
                      correct_bias=True)
    schedular = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_proportion'], t_total=t_total)

    # Main Loop
    final_model_report, best_model_report = main_loop(model, optimizer, schedular, train_dataloader, eval_dataloader,
                                                      args, logger, processor.ix2label, Smatrix,
                                                      num_epocs=args['num_train_epochs'])


    if is_final:
        # model_state_dict = torch.load(args["output_model_file"])
        model_state_dict = torch.load(args["output_model_file"] + '.final')
        model = TransformerModel(args)
        model.load_state_dict(model_state_dict)
        model.to(device)
        result, all_logits, all_labels, ix2label = evaluate(model, eval_dataloader, args, logger, processor.ix2label,
                                                            Smatrix, is_final=True)
        all_probs = pd.DataFrame(numpy_sigmoid(all_logits), columns=val_df.columns[1:])
        all_probs['Input'] = val_df['input']
        all_probs.to_csv(params['probability_output_file'], sep=',', index=None)

    return final_model_report, best_model_report


def create_S_matrix(df):
    df = df.drop('converted_label', 1)
    labels = df.columns[1:].tolist()
    labels = [int(l) for l in labels]
    label2ix = {l: ix for ix, l in enumerate(labels)}
    majors = [l for l in labels if l % 100 == 0]
    Smatrix = np.zeros((len(labels), len(labels)))
    for label in labels:
        ix = label2ix[label]
        Smatrix[ix][ix] = 1
        if label % 100 != 0:
            major = label - (label % 100)
            ix2 = label2ix[major]
            Smatrix[ix2][ix] = 1
    Smatrix = torch.from_numpy(Smatrix)
    Smatrix = Smatrix.type(torch.float32).to(device)

    return Smatrix


def train_final_model(params):
    trn_df_ = pd.read_table(os.path.join(params['data_dir'], 'train.csv'), sep=',')
    tst_df = pd.read_table(os.path.join(params['data_dir'], 'test.csv'), sep=',')
    label_num = len(trn_df_.columns) - 1

    exp_date = get_datetime_str()
    base_output_dir = params['output_dir']
    base_model_file = params['output_model_file']
    trn_df = convert_label_for_cv(trn_df_)

    if params['use_knowledge']:
        Smatrix = create_S_matrix(trn_df)
    else:
        Smatrix = None

    split_ix = 1
    log_final_models = {'f1': {}}
    log_best_models = {'f1': {}}

    log_file_name = '{}_final.log'.format(exp_date)
    model_f_name = '{}_final.bin'.format(exp_date)

    params['output_dir'] = os.path.join(base_output_dir, log_file_name)
    params['output_model_file'] = os.path.join(base_model_file, model_f_name)
    # "output_dir": './bert_saved_models/{}_report.log'.format(exp_date),

    final_model_report, best_model_report = run_fold_experiment(params, trn_df, tst_df, Smatrix, is_final=True)

    for k in final_model_report['f1'].keys():
        if k not in log_final_models['f1']:
            log_final_models['f1'][k] = final_model_report['f1'][k]
            log_best_models['f1'][k] = best_model_report['f1'][k]
        else:
            for k2 in final_model_report['f1'][k]:
                log_final_models['f1'][k][k2] += final_model_report['f1'][k][k2]
                log_best_models['f1'][k][k2] += best_model_report['f1'][k][k2]

    for k in log_final_models['f1'].keys():
        for k2 in log_final_models['f1'][k].keys():
            log_final_models['f1'][k][k2] = float("{:.2f}".format(log_final_models['f1'][k][k2]))
            log_best_models['f1'][k][k2] = float("{:.2f}".format(log_best_models['f1'][k][k2]))

    return log_final_models, log_best_models


def new_function(params):
    df_ = pd.read_table(os.path.join(params['data_dir'], 'train.csv'), sep=',')
    label_num = len(df_.columns) - 1

    exp_date = get_datetime_str()
    base_output_dir = params['output_dir']
    base_model_file = params['output_model_file']
    df = convert_label_for_cv(df_)

    if params['use_knowledge']:
        Smatrix = create_S_matrix(df)
    else:
        Smatrix = None

    skf = StratifiedKFold(n_splits=params['fold_number'], random_state=SEED, shuffle=True)
    skf.get_n_splits()

    split_ix = 1
    log_final_models = {'f1': {}}
    log_best_models = {'f1': {}}

    for trn_ix, val_ix in skf.split(np.zeros(len(df)), df.converted_label.tolist()):
        log_file_name = '{}_split_{}.log'.format(exp_date, split_ix)
        model_f_name = '{}_split_{}.bin'.format(exp_date, split_ix)

        params['output_dir'] = os.path.join(base_output_dir, log_file_name)
        params['output_model_file'] = os.path.join(base_model_file, model_f_name)
        # "output_dir": './bert_saved_models/{}_report.log'.format(exp_date),
        val_df = df.iloc[val_ix]
        trn_df = df.iloc[trn_ix]
        final_model_report, best_model_report = run_fold_experiment(params, trn_df, val_df, Smatrix)
        split_ix += 1

        for k in final_model_report['f1'].keys():
            if k not in log_final_models['f1']:
                log_final_models['f1'][k] = final_model_report['f1'][k]
                log_best_models['f1'][k] = best_model_report['f1'][k]
            else:
                for k2 in final_model_report['f1'][k]:
                    log_final_models['f1'][k][k2] += final_model_report['f1'][k][k2]
                    log_best_models['f1'][k][k2] += best_model_report['f1'][k][k2]

    for k in log_final_models['f1'].keys():
        for k2 in log_final_models['f1'][k].keys():
            log_final_models['f1'][k][k2] = float(
                "{:.2f}".format(log_final_models['f1'][k][k2] / params['fold_number']))
            log_best_models['f1'][k][k2] = float("{:.2f}".format(log_best_models['f1'][k][k2] / params['fold_number']))

    return log_final_models, log_best_models


def main_final():
    if len(sys.argv) < 3:
        sys.exit(1)

    experiments = {'params': [], 'reports': []}

    output_file = sys.argv[2]
    params = {'data_dir': '../data/',
              'output_dir': './',
              'output_model_file': './',
              'probability_output_file': output_file,
              'fold_number': 5,
              'bs': 16,
              'lr': 5e-5,
              'epoch': 20,
              'warmup_proportion': 0.1,
              'hs': 768,
              'dp': 0.3,
              'max_seq_len': 200,
              'use_knowledge': sys.argv[1].lower() == 'hle',
              }

    report = train_final_model(params)
    print(report)


if __name__ == '__main__':
    # main()
    main_final()
