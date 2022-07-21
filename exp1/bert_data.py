import pdb
import pandas as pd
import os 
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class HcsMultiLabelTextProcessor(DataProcessor):
        
    #def __init__(self,label_list):
    #    self.labels = label_list

    def __init__(self,trn_df,val_df):
        if 'converted_label' in trn_df.columns:
            df = trn_df.drop('converted_label',1)
        else:
            df = trn_df
        labels = df.columns[1:]
        label2ix = {int(x):ix for ix,x in enumerate(labels)}
        ix2label = {ix:int(x) for ix,x in enumerate(labels)}
        self.labels = labels
        self.label2ix = label2ix
        self.ix2label = ix2label
    

    def get_train_examples(self, data_df, size=-1):
        return self._create_examples(data_df,"train")
    
        
    def get_dev_examples(self, data_df, size=-1):
        return self._create_examples(data_df,"dev")
    
    def get_test_examples(self, data_df,size=-1):
        return self._create_examples(data_df,"test")        


    def _create_examples(self,data_df,set_type,labels_available=True):
        examples = []
        cntr = 0
        if 'converted_label' in data_df.columns:
            data_df = data_df.drop('converted_label',1)
        for ix,row in data_df.iterrows():
            guid =  "%s-%s" % (set_type, cntr)
            text_a = row.input
            if labels_available:
                labels_bin = row[1:].tolist()
            else:
                labels_bin = []
            cntr+=1
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels_bin))
        return examples
            


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        labels_ids = []
        for label in example.labels:
            labels_ids.append(float(label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=labels_ids))
    return features
