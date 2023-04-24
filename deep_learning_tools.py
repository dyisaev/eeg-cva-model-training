import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

from sklearn.model_selection import LeaveOneGroupOut,train_test_split
from sklearn.metrics import roc_auc_score,classification_report,accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import calibration_curve
from scipy.signal import medfilt

import pandas as pd
import pickle
import random
import os
from evaluation import  prec_rec_kappa_per_subject,cohens_kappa
from temperature_scaling import ModelWithTemperature
from temperature_scaling import CombinedLoss

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def random_sampling(model,x_test,y_test,num=256,n_bins=10,conf_sampling=None):
    probs=torch.squeeze(F.softmax(model(x_test),dim=1))
    confidences = probs[:,-1]

    sample_weights=torch.full_like(confidences,fill_value=1)
    sample_idx=torch.multinomial(sample_weights,num)
    print('y_test: ', torch.sum(y_test[sample_idx],0))
    mask_for_idx=torch.ones(sample_weights.shape)
    mask_for_idx[sample_idx]=0
    mask_for_idx=mask_for_idx==1

    return x_test[sample_idx],y_test[sample_idx],x_test[mask_for_idx],y_test[mask_for_idx],probs[sample_idx]
def sequential_sampling(model,x_test,y_test,num=256,n_bins=10,conf_sampling=None):
    probs=torch.squeeze(F.softmax(model(x_test),dim=1))
    confidences = probs[:,-1]

    sample_idx=torch.arange(0,num)
    print('y_test: ', torch.sum(y_test[sample_idx],0))
    mask_for_idx=torch.ones(confidences.shape)
    mask_for_idx[sample_idx]=0
    mask_for_idx=mask_for_idx==1
    return x_test[sample_idx],y_test[sample_idx],x_test[mask_for_idx],y_test[mask_for_idx],probs[sample_idx]

def preprocess_without_na(df,OHEncoder,save_preprocess_params,infer=False,chunks=None,keep_dataframe=False,use_ear=False):
    feat_list=['subj_id', 'frame', 'gazex', 'gazey','yaw', 'pitch', 'roll','nose_x', 'nose_y', 'ear', 
            'annot_ht','annot_gos'] 
    if chunks is not None:
        frames_arr=[]
        for it,row in chunks.iterrows():
            frames=pd.DataFrame([(row['subj_id'],frame) for frame in range(row['start_frame'],row['end_frame']+1)],columns=['subj_id','frame'])
            frames_arr.append(frames)
        df_frames=pd.concat(frames_arr,axis=0).reset_index()[['subj_id','frame']]        
        df_ml=pd.merge(df_frames,df[feat_list],how='left',left_on=['subj_id','frame'],right_on=['subj_id','frame'])
    else:
        df_ml=df[feat_list]


    if infer:
        gazex_median,gazey_median,yaw_median,pitch_median,roll_median,nosex_median,nosey_median,ear_median,OHEncoder = \
            pickle.load(open(save_preprocess_params,'rb'))        
    else:
        gazex_median=df_ml['gazex'].median()
        gazey_median=df_ml['gazey'].median()
        yaw_median=df_ml['yaw'].median()
        pitch_median=df_ml['pitch'].median()
        roll_median=df_ml['roll'].median()
        nosex_median=df_ml['nose_x'].median()
        nosey_median=df_ml['nose_y'].median()
        ear_median=df_ml['ear'].median()

    df_ml['gazex_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['gazex']-gazex_median)
    df_ml['gazex_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['gazex']+gazex_median)


    df_ml['gazey_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['gazey']-gazey_median)
    df_ml['gazey_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['gazey']+gazey_median)


    df_ml['yaw_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['yaw']-yaw_median)
    df_ml['yaw_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['yaw']+yaw_median)

    df_ml['pitch_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['pitch']-pitch_median)
    df_ml['pitch_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['pitch']+pitch_median)

    df_ml['roll_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['roll']-roll_median)
    df_ml['roll_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['roll']+roll_median)

    df_ml['nosex_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['nose_x']-nosex_median)
    df_ml['nosex_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['nose_x']+nosex_median)

    df_ml['nosey_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['nose_y']-nosey_median)
    df_ml['nosey_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['nose_y']+nosey_median)

    # remove NA labels and create target variable
    df_ml['annot_gos'][df_ml['annot_gos'].isna()]=-10
    df_ml['annot_ht'][df_ml['annot_ht'].isna()]=-10

    df_ml_filtered=df_ml[np.logical_and(df_ml['annot_gos']!=-10,df_ml['annot_ht']!=-10)]
    df_ml_filtered['y_target']=np.logical_or(df_ml['annot_gos'],df_ml['annot_ht']).astype('float')

    
    if keep_dataframe==False:
    # create data, and add one-hot coding of the subject ID
        data=df_ml_filtered[['gazex_plus', 'gazex_minus','gazey_plus', 'gazey_minus','yaw_plus', 'yaw_minus','pitch_plus', 'pitch_minus','roll_plus', 'roll_minus',
                'nosex_plus', 'nosex_minus','nosey_plus', 'nosey_minus','ear', 'y_target']] if use_ear else df_ml_filtered[['gazex_plus', 'gazex_minus','gazey_plus', 'gazey_minus','yaw_plus', 'yaw_minus','pitch_plus', 'pitch_minus','roll_plus', 'roll_minus',
                'nosex_plus', 'nosex_minus','nosey_plus', 'nosey_minus','y_target']]
        data = data.to_numpy()
        # add one-hot encoding for subject ID
        if OHEncoder is not None:
            if not infer:
                OHEncoder=OneHotEncoder(sparse=False).fit(df_ml_filtered[['subj_id']])
                oh=OHEncoder.transform(df_ml_filtered[['subj_id']])
                id_onehot=np.concatenate([oh,np.zeros((oh.shape[0],1))],axis=-1)
            else:
                oh_infer=np.zeros((1,OHEncoder.categories_[0].shape[0]+1))
                oh_infer[0,-1]=1
                id_onehot=np.repeat(oh_infer,df_ml_filtered.shape[0],axis=0)
                print('shape,sum: ',id_onehot.shape,np.sum(id_onehot[:,-1]))
#            id_onehot=OHEncoder.transform(df_ml_filtered[['subj_id']])
            data=np.concatenate((id_onehot,data),axis=1)
    else:
        data=df_ml_filtered[['gazex_plus', 'gazex_minus','gazey_plus', 'gazey_minus','yaw_plus', 'yaw_minus','pitch_plus', 'pitch_minus','roll_plus', 'roll_minus',
                'nosex_plus', 'nosex_minus','nosey_plus', 'nosey_minus', 'ear','frame','subj_id',
                'y_target']] if use_ear else df_ml_filtered[['gazex_plus', 'gazex_minus','gazey_plus', 'gazey_minus','yaw_plus', 'yaw_minus','pitch_plus', 'pitch_minus','roll_plus', 'roll_minus',
                'nosex_plus', 'nosex_minus','nosey_plus', 'nosey_minus', 'frame','subj_id',
                'y_target']]
        if OHEncoder is not None:
            if not infer:
                OHEncoder=OneHotEncoder(sparse=False).fit(df_ml_filtered[['subj_id']])
                oh=OHEncoder.transform(df_ml_filtered[['subj_id']])
                id_onehot=np.concatenate([oh,np.zeros((oh.shape[0],1))],axis=-1)
#            id_onehot=OHEncoder.transform(df_ml_filtered[['subj_id']])
            else:
                oh_infer=np.zeros(OHEncoder.categories_[0].shape[0]+1)
                oh_infer[-1]=1
                id_onehot=np.repeat([oh_infer],df_ml_filtered.shape[0],axis=-1)
                print('shape,sum: ',id_onehot.shape,np.sum(id_onehot[:,-1]))

            df_onehot=pd.DataFrame(id_onehot)
            df_onehot.index=df_ml_filtered.index          
            data=pd.concat([df_onehot,data],axis=1)
    if not infer:
        params=[gazex_median,gazey_median,yaw_median,pitch_median,roll_median,nosex_median,nosey_median,ear_median,OHEncoder]
        pickle.dump(params,open(save_preprocess_params,'wb'))

    return df_ml_filtered[['frame']],data



class MLP_2HiddenLayers(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_2HiddenLayers, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 14),
            nn.ReLU(),
        )
        self.lastlayer= nn.Linear(14, output_dim)

    def forward(self, x):

        logits = self.lastlayer(self.linear_relu_stack(x))
        return logits


def run_activelearning_test_with_saving(df,model_name,preprocess,preprocess_params,combined_loss_param=0.5,batch_size=1024,N_epochs=5,learning_rate=0.001,N_ALepochs=20,half_batch_size_AL=16,
                    saved_models_path=f'/media/st4Tb/projects/eeg-inattention/saved_models/',use_ear=False, sampling_type=random_sampling,sbj_to_save=[]):
    loocv=LeaveOneGroupOut()
    group=df[['subj_id']].to_numpy()[:,0]

    #learning_rate = 0.001
    #batch_size=1024
    #N_epochs=5
    auc_test_before_ts_arr=[]
    auc_test_arr=[]
    auc_test_after_al_arr=[]
    idx=-1
    patients_arr=[]
    #model_name='model_both'
    res=[]
    coh=[]
    for trainval,test in loocv.split(df,groups=group):
        idx+=1
        patient_id=np.unique(group[test])

        df_trainval=df.iloc[trainval,:]
        frames_trainval,data_trainval=preprocess_without_na(df_trainval,preprocess_params[0],preprocess_params[1]+f'{model_name}_{patient_id[0]}.pickle',infer=True, use_ear = use_ear)
        pi_1=np.mean(data_trainval[:,-1])
        print(f'patient {patient_id} pos class prevalence: {pi_1}')    

        df_test=df.iloc[test,:]
        
        frames_test,data_test=preprocess_without_na(df_test,preprocess_params[0],preprocess_params[1]+f'{model_name}_{patient_id[0]}.pickle',infer=True, use_ear = use_ear)
        x_test=data_test[:,:-1]
        y_test=data_test[:,-1]
        y_test=np.stack([(y_test==0).astype(int),(y_test!=0).astype(int)],axis=1)
        
        
        x_test=torch.FloatTensor(x_test)   #.to(device)
        y_test=torch.Tensor(y_test.astype(int))  #.to(device)
        testDataset = torch.utils.data.TensorDataset(x_test,y_test)


        scaled_model=torch.load(saved_models_path+f'Scaled_{model_name}_{patient_id[0]}.pt')

        transfer_optim = torch.optim.Adam(scaled_model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(transfer_optim, step_size=10, gamma=0.8)

        auc_t_al_arr=[]
        loss_t_al_arr=[]
        al_criterion=CombinedLoss(combined_loss_param,n_bins=16)


     #   N_ALepochs=20
        loss_t_al_arr=[]
        x_leftovers=x_test
        y_leftovers=y_test

        for epoch in range(N_ALepochs):     #.model.linear_relu_stack

            predicted_test_al=torch.squeeze(F.softmax(scaled_model(x_test.to(device)),dim=1))
            
#            auc_test_al=roc_auc_score(y_test.cpu().numpy(),predicted_test_al.detach().cpu().numpy())
            
            pred_model=predicted_test_al.detach().cpu().numpy()[:,1]
            gt = y_test.cpu().numpy()[:,1]
            
            pred_model=medfilt(pred_model,11)
            auc_test_al=roc_auc_score(gt,pred_model)

            if (patient_id[0] in sbj_to_save) and (epoch in [0,5,10,20]):
                torch.save(scaled_model,saved_models_path+f'SUBJ_{patient_id[0]}_epoch{epoch}_{sampling_type.__name__}_model.pt')
                pickle.dump([pred_model,gt],open(saved_models_path+f'SUBJ_{patient_id[0]}_epoch{epoch}_{sampling_type.__name__}_pred.pickle','wb'))
            precision,recall,thresholds,avg_prec,auc=prec_rec_kappa_per_subject(pred_model,gt)
            res_df=pd.DataFrame([(patient_id[0],epoch,auc,avg_prec)],columns=['subj_id','epoch','auc','avg_prec'])
            coh_kap_thr,coh_kap_val=cohens_kappa(pred_model,gt,num=501)
            coh_df=pd.DataFrame([(coh_thr,coh_val) for (coh_thr, coh_val) in zip(coh_kap_thr,coh_kap_val)],columns=['threshold','kappa'])
            coh_df_max=coh_df.loc[coh_df['kappa'].idxmax()][['threshold','kappa']]
            res_df['kappa_thr']=coh_df_max['threshold']
            res_df['kappa']=coh_df_max['kappa']
            coh_df['subj_id']=patient_id[0]
            coh_df['epoch']=epoch
            coh.append(coh_df)                  
            print(f'patient: {patient_id}; epoch:{epoch}; AUC AL: {auc_test_al}')
            if epoch % 20 ==0:
                res_df.to_csv(f'{saved_models_path}res_MLP_batch64_activeLearnRot20_{sampling_type.__name__}_{patient_id[0]}_ep{epoch}.csv')
                #pickle.dump(res_df,open(f'{saved_models_path}res_MLP_batch64_activeLearn_{patient_id[0]}_ep{epoch}.pickle','wb'))
            if x_leftovers.shape[0]<int(half_batch_size_AL*2):
                print (f'epoch {epoch}, no data left for sampling without replacement')
                break
            conf_sampling = epoch>=12
            if epoch>0:
                x_prev=x_t
                y_prev=y_t     

                
                x_t,y_t,x_leftovers,y_leftovers,prob_samples = sampling_type(scaled_model,x_leftovers.to(device),y_leftovers.to(device),num=int(half_batch_size_AL*2),n_bins=16,conf_sampling=conf_sampling) #margin_sampling(scaled_model,x_test.to(device),y_test.to(device),num=1024) #lc_sampling(model,x_test.to(device),y_test.to(device),num=1024)#repr_sampling(scaled_model,x_test.to(device),y_test.to(device),num=int(half_batch_size_AL*2),n_bins=16) #margin_sampling(scaled_model,x_test.to(device),y_test.to(device),num=1024) #lc_sampling(model,x_test.to(device),y_test.to(device),num=1024)
                observed_pos_frequency, mean_pred_proba  = calibration_curve(y_t.detach().cpu().numpy()[:,1], prob_samples.detach().cpu().numpy()[:,1], n_bins=10, strategy='uniform')

                x_t=torch.concat((x_prev,x_t),axis=0)
                y_t=torch.concat((y_prev,y_t),axis=0)

                print(f'sum y_t: {y_t.sum(axis=0)}')

            else:                
                x_t,y_t,x_leftovers,y_leftovers,prob_samples = sampling_type(scaled_model,x_leftovers.to(device),y_leftovers.to(device),num=int(half_batch_size_AL*2),n_bins=16,conf_sampling=conf_sampling)#repr_sampling(scaled_model,x_test.to(device),y_test.to(device),num=int(half_batch_size_AL*2),n_bins=16) #margin_sampling(scaled_model,x_test.to(device),y_test.to(device),num=1024) #lc_sampling(model,x_test.to(device),y_test.to(device),num=1024)
                observed_pos_frequency, mean_pred_proba  = calibration_curve(y_t.detach().cpu().numpy()[:,1], prob_samples.detach().cpu().numpy()[:,1], n_bins=10, strategy='uniform')

            print('leftovers shapes: ',x_leftovers.shape,y_leftovers.shape)
            loss_not_decreased=0

            x_res=x_t
            y_res=y_t



            res_df['amount_positive']=np.sum((y_res.detach().cpu().numpy()[:,1]==1).astype(int))
            res_df['observed_pos_freq']=[observed_pos_frequency]
            res_df['mean_pred_proba']=[mean_pred_proba]

            res.append(res_df)    
            print(x_res.detach().cpu().numpy().shape,(y_res.detach().cpu().numpy()[:,1]==1).astype(int).shape,np.sum((y_res.detach().cpu().numpy()[:,1]==1).astype(int)))
            print('obs-pos-freq: ', observed_pos_frequency)
            print('mean-pred-proba: ', mean_pred_proba)

            for rotation in range (20):


                outputs_t = scaled_model(x_res)
                loss = al_criterion(torch.squeeze(outputs_t), y_res) 
                transfer_optim.zero_grad() 
                loss.backward()
                transfer_optim.step() 

            scheduler.step()
    return [res,coh]




def run_experiment_1(df,model_name,preprocess,preprocess_params,combined_loss_param=0.5,batch_size=1024,N_epochs=5,learning_rate=0.001,N_ALepochs=20,half_batch_size_AL=16,
                    saved_models_path=f'/media/st4Tb/projects/eeg-inattention/saved_models/',use_ear=False):


    loocv=LeaveOneGroupOut()
    group=df[['subj_id']].to_numpy()[:,0]

    auc_test_before_ts_arr=[]
    auc_test_arr=[]
    auc_test_after_al_arr=[]
    idx=-1
    patients_arr=[]
    for trainval,test in loocv.split(df,groups=group):
        idx+=1
        patient_id=np.unique(group[test])

        frames,data_preprocessed=preprocess_without_na(df.iloc[trainval,:],preprocess_params[0],preprocess_params[1]+f'{model_name}_{patient_id[0]}.pickle',infer=False, use_ear = use_ear) 
        data_train,data_val=train_test_split(data_preprocessed,stratify=data_preprocessed[:,-1],random_state=42)

        df_test=df.iloc[test,:]
        
        frames_test,data_test=preprocess_without_na(df_test,preprocess_params[0],preprocess_params[1]+f'{model_name}_{patient_id[0]}.pickle',infer=True, use_ear = use_ear)

        patient_id=np.unique(group[test])
        patients_arr.append(patient_id[0])
        x_train=data_train[:,:-1]
        y_train=data_train[:,-1] 
        y_train=np.stack([(y_train==0).astype(int),(y_train!=0).astype(int)],axis=1)

        x_val=data_val[:,:-1]
        y_val=data_val[:,-1]
        y_val=np.stack([(y_val==0).astype(int),(y_val!=0).astype(int)],axis=1)

        x_test=data_test[:,:-1]
        y_test=data_test[:,-1]
        y_test=np.stack([(y_test==0).astype(int),(y_test!=0).astype(int)],axis=1)

        count=np.sum(y_train,axis=0)
        class_count=np.array([count[0],count[1]])
        weight=1./class_count
        samples_weight = np.array([weight[np.argmax(t)] for t in y_train])
        samples_weight=torch.from_numpy(samples_weight).to(device)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        count_val = np.sum(y_val,axis=0)
        class_count_val = np.array([count_val[0],count_val[1]])
        weight_val=1./class_count_val
        samples_weight_val = np.array([weight_val[np.argmax(t)] for t in y_val])
        samples_weight_val = torch.from_numpy(samples_weight_val).to(device)
        sampler_val = torch.utils.data.WeightedRandomSampler(samples_weight_val, len(samples_weight_val))


        x_train=torch.FloatTensor(x_train)
        y_train=torch.Tensor(y_train.astype(int))

        x_val=torch.FloatTensor(x_val)
        y_val=torch.Tensor(y_val.astype(int))

        x_test=torch.FloatTensor(x_test)   
        y_test=torch.Tensor(y_test.astype(int)) 

        trainDataset = torch.utils.data.TensorDataset(x_train,y_train)
        validDataset = torch.utils.data.TensorDataset(x_val,y_val)
        testDataset = torch.utils.data.TensorDataset(x_test,y_test)


        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=batch_size, num_workers=1, sampler = sampler)
        valLoader = torch.utils.data.DataLoader(dataset = validDataset, batch_size=batch_size, num_workers=1, sampler = sampler_val)

        input_dim = data_preprocessed.shape[1]-1 # Two inputs x1 and x2 
        output_dim = 2 # Two possible outputs


        model = MLP_2HiddenLayers(input_dim,output_dim)
        model = ModelWithTemperature(model).to(device)
    #    scaled_model.set_temperature(valLoader)

    #    print(model)
        criterion = torch.nn.CrossEntropyLoss() #CombinedLoss(0.0,n_bins=45) # torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=0.001) #SGD(model.parameters(), lr=learning_rate,momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        best_auc_val=-1
        for epoch in range(N_epochs):

            losses = []
            losses_test = []
            Iterations = []
            it = 0
            for it, (x,y) in enumerate(trainLoader):
                x=x.to(device)
                y=y.to(device)
                outputs = model(x)
                loss = criterion(outputs,y)#criterion(torch.squeeze(outputs), y) # [200,1] -squeeze-> [200]
                optimizer.zero_grad() # Setting our stored gradients equal to zero
                loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
                optimizer.step() # Updates weights and biases with the optimizer (SGD)
                if it%1000==0:
                    # calculate AUC
                    with torch.no_grad():

                        predicted_train=torch.squeeze(outputs).detach().cpu().numpy()
                        auc_train=roc_auc_score(y.detach().cpu().numpy(), predicted_train)

                        outputs_val = torch.squeeze(model(x_val.to(device)))
                        loss_val = criterion(outputs_val, y_val.to(device))
                        predicted_val = outputs_val.detach().cpu().numpy()

                        auc_val=roc_auc_score(y_val.detach().cpu().numpy(),predicted_val)

                        #model.set_temperature(valLoader)

                        if    auc_val > best_auc_val: 
                            torch.save(model,saved_models_path+f'{model_name}_{patient_id[0]}.pt')
                            best_auc_val=auc_val

                        print(f"Epoch: {epoch}; Iteration: {it}. \nVal - Loss: {loss_val.item()}. AUC: {auc_val}")
                        print(f"Train -  Loss: {loss.item()}. AUC: {auc_train}\n")
            scheduler.step()
        model=torch.load(saved_models_path+f'{model_name}_{patient_id[0]}.pt')
        predicted_test0=torch.squeeze( F.softmax(model(x_test.to(device)),dim=1))
        try:
            auc_test_before_ts_arr.append(roc_auc_score(y_test.cpu().numpy(),predicted_test0.detach().cpu().numpy()))
        except:
            print(f'patient: {patient_id}. Exception. Accuracy=',accuracy_score(y_test.cpu().numpy()[:,1],predicted_test0.detach().cpu().numpy()[:,1]>0.5))
        print(f'patient: {patient_id}; AUC: {auc_test_before_ts_arr}')

        model.set_temperature(valLoader)
        scaled_model=model
        torch.save(scaled_model,saved_models_path+f'Scaled_{model_name}_{patient_id[0]}.pt')
    df_new=pd.DataFrame(zip(patients_arr,auc_test_arr,auc_test_after_al_arr),columns=['ID','AUC_TS','AUC_TS_AL'])
    df_new.to_csv(saved_models_path+f'{model_name}_AUC.csv')
