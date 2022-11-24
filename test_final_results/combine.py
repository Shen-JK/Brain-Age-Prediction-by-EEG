import os
import pandas as pd
import numpy as np

f1=open('melspec_class_equalprob_average_fold1_20.csv','r')
lines1=f1.readlines()
f2=open('melspec_class_equalprob_average_fold1_30.csv','r')
lines2=f2.readlines()
f3=open('melspec_class_equalprob_average_fold1_36.csv','r')
lines3=f3.readlines()
f4=open('melspec_class_equalprob_average_fold2_27.csv','r')
lines4=f4.readlines()
f5=open('melspec_class_equalprob_average_fold2_40.csv','r')
lines5=f5.readlines()
f6=open('melspec_class_equalprob_average_fold2_80.csv','r')
lines6=f6.readlines()
f7=open('melspec_class_equalprob_average_fold3_17.csv','r')
lines7=f7.readlines()
f8=open('melspec_class_equalprob_average_fold3_32.csv','r')
lines8=f8.readlines()
f9=open('melspec_class_equalprob_average_fold3_42.csv','r')
lines9=f9.readlines()
f10=open('melspec_class_equalprob_average_fold6_15.csv','r')
lines10=f10.readlines()
f11=open('melspec_class_equalprob_average_fold6_16.csv','r')
lines11=f11.readlines()
f12=open('melspec_class_equalprob_average_fold6_39.csv','r')
lines12=f12.readlines()


pred_lis=[]
for i in range(1,len(lines1)):
    # print(lines1[i])
    pred_lis.append(
                (float(lines1[i].split(',')[-1][:-1])+float(lines2[i].split(',')[-1][:-1]) \
                +float(lines3[i].split(',')[-1][:-1])+float(lines4[i].split(',')[-1][:-1]) \
                +float(lines5[i].split(',')[-1][:-1])+float(lines6[i].split(',')[-1][:-1]) \
                +float(lines7[i].split(',')[-1][:-1])+float(lines8[i].split(',')[-1][:-1]) \
                +float(lines9[i].split(',')[-1][:-1])+float(lines10[i].split(',')[-1][:-1]) \
                +float(lines11[i].split(',')[-1][:-1])+float(lines12[i].split(',')[-1][:-1])
                )/12)
    
    
    
dummy_submission = []
for subj, pred in zip(range(1601, 1601 + 400), np.array(pred_lis)):
    dummy_submission.append({"id": subj, "age": pred})
pd.DataFrame(dummy_submission).to_csv("mysubmission_combineMITfold1236_final_re.csv", index=False)