import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sys
from deep_learning_tools import run_experiment_1,preprocess_without_na
from deep_learning_tools import set_seed

import pickle


set_seed(1296)

df=pd.read_csv('dataset-deidentified.csv')
OHEncoder=True
save_preprocess_fname='sandbox/MLP_2HL_SubmissionMar16_Preproc'
preprocess_params=[OHEncoder,save_preprocess_fname]
model_name='MLP_2HL_SubmissionMar16'
combined_loss_param=0.0
run_experiment_1(df,model_name,preprocess_without_na,preprocess_params,saved_models_path=f'sandbox/',combined_loss_param=combined_loss_param,N_epochs=5,N_ALepochs=0,half_batch_size_AL=64,use_ear=False)
