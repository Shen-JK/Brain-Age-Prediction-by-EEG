# Brain-Age-Prediction-by-EEG
Code for brain age prediction from EEG in challenge https://codalab.lisn.upsaclay.fr/competitions/8336# 

## Folder directory

- dataset  (training and testing files in .scp format) 
```
dataset
│
└───fold1
│   └───train.scp  (1000 subjects in training dataset, with original float ages as labels)
│   └───train_class.scp  (1000 subjects in training dataset, with int ages as labels by rounding float brain ages)
│   └───valid.scp  (100 subjects in training dataset, with original float ages as labels)
│   └───valid_class.scp  (100 subjects in training dataset, with int ages as labels by rounding float brain ages)
│   └───test.scp  (100 subjects in training dataset, with original float ages as labels)
|   └───test_class.scp  (100 subjects in training dataset, with int ages as labels by rounding float brain ages)
└───fold2
└───fold3
└───fold4
└───fold5
└───fold6
└───testing_flat.scp  (400 subjects in development phase)
└───test_final.scp  (400 subjects in final test phase)
└───train_class.scp  (1200 subjects in training dataset)
└───train_subjects.csv  (brain ages  of 1200 subjects in training dataset)
```
- log  (For reproducing the training process, this folder is used to save the checkpoints)
- test_final  (The unzip .fif files in final test phase)
```
test_final
└───subj2001_EC_raw.fif
└───subj2001_EO_raw.fif
└───subj2002_EC_raw.fif
└───subj2002_EO_raw.fif
└───......
```
- test_final_results (To get the ensembled results)
```
test_final_results
└───combine.py
└───.csv
```
- training  (The unzip .fif files in training dataset)
```
training
└───subj0001_EC_raw.fif
└───subj0001_EO_raw.fif
└───......
└───subj1200_EC_raw.fif
└───subj1200_EO_raw.fif
```
- load_data.py  (dataloader code)
- model_use.py  (network code)
- run_train_MIT.sh 
- train_MIT.py  (training code)

## Main idea

We change this regression problem to classification problem. The brain ages in training set are ranging from 4 to 24. The float ages are rounded so this question becomes to classify 21 classes. To overcome overfitting, an offset of 1,0,-1 is added to ages in training stage. The max two probabilities are averaged to get the prediction results in test stage. The 1200 subjects in training set are divided into six parts and the models are ensembled. As fold4 and fold5 show no performance improvement on development phase, the models of the other four folds are ensembled to get final results.

run_train_MIT.sh shows some hyper-parameters in training stage.

train_MIT.py shows the training (function train_spec), testing (function valid_class or test_class), and final test (function final_test_class) code.

load_data.py shows labels, offsets and feature extraction. The melspectrograms of EC and EO of each subjects are extracted.

model_use.py shows the network structure. The features are input into the network as follows
```
       melspec of EC           melspec of EO
         1*153*128               1*75*128
         ____|____              ____|____
        |         |            |         |
      1*153*64  1*153*64    1*75*64   1*75*64
        |         |            |         |
    branch 1a  branch 1b   branch 2a  branch 2b
        |_________|            |_________|
             |______________________|
                          |
                       21 classes
```
## Procedures to reproduce training and testing

The model is trained on single NVIDIA Tesla P100, with torch1.7.1+cu101.

First, put the 1200 EC and EO into training folder. And put the 400 EC and EO in final test into test_final folder.

Then, run the following code in turn, and the models will be saved in log folder.

./run_train_MIT.sh 1 fold1

./run_train_MIT.sh 1 fold2

./run_train_MIT.sh 1 fold3

./run_train_MIT.sh 1 fold6

To get the test results, run

./run_train_MIT.sh 0 fold1

./run_train_MIT.sh 0 fold2

./run_train_MIT.sh 0 fold3

./run_train_MIT.sh 0 fold6

cd ./test_final_results

python combine.py

and the '*mysubmission_combineMITfold1236_final_re.csv'* will be generated.

## Procedures to reproduce testing only  by using trained models

If you just want to reproduce the test process, you can use our trained model in log_for_test. Download the log_for_test.rar in trained_models in Releases. Unzip it and change the folder name'log_for_test' to 'log' and run

./run_train_MIT.sh 0 fold1

./run_train_MIT.sh 0 fold2

./run_train_MIT.sh 0 fold3

./run_train_MIT.sh 0 fold6

cd ./test_final_results

python combine.py
