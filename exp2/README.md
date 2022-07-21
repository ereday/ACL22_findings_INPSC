# Experiment 2: Party Manifestos

## Dependencies

 ```
 cd src
 pip install -r requirements.txt
 ```

## Manifesto Data

 We are not allowed to redistribute the data that we used in our experiments [1]. However, you can use data_download.sh to download and generate data splits:

 ```
 bash data_download.sh
 ```

 [1] https://manifesto-project.wzb.eu/information/terms_of_use


## Pretrained BERT models

For each language,  we  use  a  cased  BERT  variant that was trained specifically for the target language:

    Fi:https://github.com/TurkuNLP/FinBERT
    De:https://deepset.ai/german-bert
    Hu:https://hlt.bme.hu/en/resources/hubert
    Tr:https://github.com/dbmdz/berts
    En:https://huggingface.co/bert-base-cased

Please download them and locate under pretrained_bert_models/ directory.


## How to run the code

Please use the ```src/run.sh``` script for training the models.
