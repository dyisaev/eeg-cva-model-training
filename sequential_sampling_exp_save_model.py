import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sys
sys.path.append('../../tools')
from deep_learning_tools import run_activelearning_test_with_saving,preprocess_without_na,repr_sampling, sequential_sampling
from deep_learning_tools import set_seed
import pickle

set_seed(1296)

df=pd.read_csv('dataset-deidentified.csv')
save_preprocess_fname='../../saved_models/MLP_2HL_SubmissionMar16_Preproc'
preprocess_params=[True,save_preprocess_fname]
model_name='MLP_2HL_SubmissionMar16'
combined_loss_param=0.0
al_res=run_activelearning_test_with_saving(df,model_name,preprocess_without_na,preprocess_params,saved_models_path=f'sandbox/',combined_loss_param=combined_loss_param,
                    N_epochs=5,N_ALepochs=50,half_batch_size_AL=64,use_ear=False, sampling_type=sequential_sampling,sbj_to_save=[])     

pickle.dump(al_res,open(f'sequential_sampling_res_savemodel_Mar16.pickle','wb'))
