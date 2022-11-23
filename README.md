# Brain-Age-Prediction-by-EEG
Code for brain age prediction from EEG in challenge https://codalab.lisn.upsaclay.fr/competitions/8336# 

Folder directory

-dataset  (training and testing files in .scp format) 
  -fold1
    train.scp  (1000 subjects in training dataset, with original float ages as labels)
    train_class.scp  (1000 subjects in training dataset, with int ages as labels by rounding float brain ages)
    valid.scp  (100 subjects in training dataset, with original float ages as labels)
    valid_class.scp  (100 subjects in training dataset, with int ages as labels by rounding float brain ages)
    test.scp  (100 subjects in training dataset, with original float ages as labels)
    test_class.scp  (100 subjects in training dataset, with int ages as labels by rounding float brain ages)
    
  -fold2
  
  -fold3
  
  -fold4
  
  -fold5
  
  -fold6
  
  testing_flat.scp  (400 subjects in development phase)
  test_final.scp  (400 subjects in final test phase)
  train_class.scp  (1200 subjects in training dataset)
  train_subjects.csv  (brain ages  of 1200 subjects in training dataset)
