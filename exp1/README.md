# Experiment 1: Newspapers

## Dependencies & Installation
- Note that the code for ILP decoding relies on a python library called 'PySCIPOpt' which requires a working installation of the SCIP Optimization Suite. Please, make sure that your SCIP installation works before running the command below.

 ```
 pip install -r requirements.txt
 ```

Please make sure the dataset is placed under `data` folder:

```
mv data/train.csv ./exp1/data/.
mv data/test.csv ./exp1/data/.
cd exp1/src
```


## To train the models:
```
# Base model:
CUDA_VISIBLE_DEVICES=0 python bert_main_wo_reg.py plain bert_plain_test_output.csv 0.3 5e-5
# HLE Model:
CUDA_VISIBLE_DEVICES=0 python bert_main_wo_reg.py hle bert_hle_test_output.csv 0.3 5e-5
# CRR Model:
CUDA_VISIBLE_DEVICES=0 python bert_main.py plain bert_crr_test_output.csv 0.005 -0.01 0.4 5e-5
# HLE+CRR Model:
CUDA_VISIBLE_DEVICES=0 python bert_main.py hle bert_hle_crr_test_output.csv 0.01 -0.01 0.4 5e-5

# ILP Model:
python ILP_decoder.py bert_plain_test_output.csv
# HLE+ILP Model:
python ILP_decoder.py bert_hle_test_output.csv   
# CRR+ILP Model:
python ILP_decoder.py bert_crr_test_output.csv   
# HLE+CRR+ILP Model:
python ILP_decoder.py bert_hle_crr_test_output.csv   
```
