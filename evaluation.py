import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score,cohen_kappa_score
from tqdm import tqdm
def prec_rec_per_event(binary_signal,start,stop):
    if stop-start>0:
        return np.sum(binary_signal[start+1:stop+1])/(stop-start)
    else:
        return 0
def average_precision_per_subject(pred,gt):
    pred_extended=np.insert(np.append(pred,0),0,0)
    gt_extended=np.insert(np.append(gt,0),0,0)
    start_markers=np.where(np.diff(pred_extended)>0)[0]
    stop_markers=np.where(np.diff(pred_extended)<0)[0]
    precisions=[]
    correct_predictions_count=0
    for start,stop in zip(start_markers,stop_markers):
        prec=prec_rec_per_event(gt_extended,start,stop)
        precisions+=[prec]
        if prec>0.5:
            correct_predictions_count+=1
    if len(precisions)>0:
        return np.mean(precisions),correct_predictions_count/len(precisions)  
    else:
        return np.nan,np.nan

def average_recall_per_subject(pred,gt):
    pred_extended=np.insert(np.append(pred,0),0,0)
    gt_extended=np.insert(np.append(gt,0),0,0)
    start_markers=np.where(np.diff(gt_extended)>0)[0]
    stop_markers=np.where(np.diff(gt_extended)<0)[0]
    recalls=[]
    correct_predictions_count=0
    for start,stop in zip(start_markers,stop_markers):
        rec=prec_rec_per_event(pred_extended,start,stop)
        recalls+=[rec]
        if rec>0.5:
            correct_predictions_count+=1
    return np.mean(recalls),correct_predictions_count/len(recalls)

def correct_prediction(binary_signal,start,stop):
    if prec_rec_per_event(binary_signal,start,stop)>0.5:
        return 1
    else:
        return 0
    
def precision_recall_f1_per_subject(pred_continuous,gt,threshold):
#    print(pred_continuous.shape,gt.shape)
    auc=roc_auc_score(gt,pred_continuous)
    pred=pred_continuous>threshold
    avg_prec,pps=average_precision_per_subject(pred,gt)
    avg_rec,rps=average_recall_per_subject(pred,gt)
#    f1=2*pps*rps/(pps+rps)
    return avg_prec,avg_rec,pps,rps,auc

def prec_rec_kappa_per_subject(pred_continuous,gt):
    precision, recall, thresholds = precision_recall_curve(gt, pred_continuous)
    avg_prec=average_precision_score(gt,pred_continuous)
    auc=roc_auc_score(gt,pred_continuous)
    return precision,recall,thresholds,avg_prec,auc
    
def cohens_kappa(pred_continuous,gt,num=1001):
    kappa=[]
    thresholds=np.linspace(0,1,num=num)
    print('thr len', len(thresholds))
    for thr in tqdm(thresholds):
        pred=(pred_continuous>thr)
        kappa.append(cohen_kappa_score(pred,gt))
    return thresholds,kappa
    