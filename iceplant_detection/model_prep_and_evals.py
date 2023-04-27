import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import fbeta_score

# **********************************************************************************************************
# **********************************************************************************************************

def test_train_aois_scenes(samples, test_size=0.3):
    all_train = []
    all_test = []

    X_labels = samples.columns.drop('iceplant')

    aois = samples.aoi.unique()

    for aoi in aois:
        in_aoi = samples[samples.aoi == aoi]    
        scenes = in_aoi.naip_id.unique()
        for scene in scenes:
            in_scene = in_aoi[in_aoi.naip_id == scene]

            X = np.array(in_scene.drop('iceplant', axis = 1))
            y = np.array(in_scene['iceplant'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size = test_size, 
                                                                random_state = 42)
            train = pd.DataFrame(X_train, columns = X_labels)
            train['iceplant'] = y_train

            test = pd.DataFrame(X_test, columns = X_labels)
            test['iceplant'] = y_test

            all_train.append(train)
            all_test.append(test)

    ignore = False
    train = pd.concat(all_train, ignore_index=ignore)
    test = pd.concat(all_test, ignore_index=ignore)
    return train, test

# ---------------------------------
def test_train_from_df(df,test_size=0.3):
    # y = values we want to predict
    y = np.array(df['iceplant'])
    # X = features
    X = np.array(df.drop('iceplant', axis = 1))
    return train_test_split(X, y, 
                            test_size = test_size, 
                            random_state = 42)

# ---------------------------------
# --- print shapes of train/test features/labels
def  train_test_shapes(train_features, train_labels, test_features, test_labels):
    print('Training Features Shape:', train_features.shape) 
    print('Training Labels Shape:', train_labels.shape) 
    print('Testing Features Shape:', test_features.shape) 
    print('Testing Labels Shape:', test_labels.shape)
    print()
    return

# --- print proportions of ice plant (1) vs no iceplant (0) in an array with only 0 and 1
def iceplant_proportions(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('no-iceplant:iceplant ratio    ',round(counts[0]/counts[1],1),':1')
    n = labels.shape[0]
    perc = [round(counts[0]/n*100,2), round(counts[1]/n*100,2)]
    df = pd.DataFrame({'iceplant':unique,
             'counts':counts,
             'percentage':perc}).set_index('iceplant')
    print(df)
    print()

# --- print proportions of ice plant vs no ice plant in train/test sets
def test_train_proportions(train_labels, test_labels):
    print('TRAIN SET COUNTS:')
    iceplant_proportions(train_labels)

    print('TEST SET COUNTS:')
    iceplant_proportions(test_labels)

    return


# **********************************************************************************************************
# **********************************************************************************************************

def accuracy_info_df(y_true, y_pred):
    N = y_true.shape[0]
    unique, counts = np.unique(y_true,return_counts=True)    
    
    confmtx = confusion_matrix(y_true, y_pred)
    TN = confmtx[0,0]
    FP = confmtx[0,1]
    FN = confmtx[1,0]
    TP = confmtx[1,1]

    # sensitivity : TP/P
    sens =  np.round( confmtx[1,1] / counts[1] * 100, 2) 

    # specificity : TN/N
    spec =  np.round( confmtx[0,0] / counts[0] * 100, 2) 

    # precision P : TP/(TP+FP)
    prec_P = np.round( confmtx[1,1] / (confmtx[1,1]+confmtx[0,1]) * 100, 2) 
    
    # precision N : TN/(TN+FN)
    prec_N = np.round( confmtx[0,0] / (confmtx[0,0]+confmtx[1,0]) * 100, 2)
    
    # overal accuracy: (TP + TN)/(P + N)
    acc = np.round( (confmtx[1,1] + confmtx[0,0])/y_true.shape[0]*100,2) 
    
    D = {'TN':TN, 'TP':TP, 'FN':FN, 'FP':FP, 
         'acc':acc, 
         'sens':sens, 'prec_P':prec_P,
         'spec':spec, 'prec_N':prec_N}
    df = pd.DataFrame([D])
    return df

# ---------------------------------

# https://stackoverflow.com/questions/61466961/what-do-the-normalize-parameters-mean-in-sklearns-confusion-matrix
# *** change to y_true / y_pred

def print_accuracy_info(y_true,y_pred):
    N = y_true.shape[0]
    
    confmtx = confusion_matrix(y_true,y_pred)
    
    print('true negatives:', confmtx[0,0], 
          '    false positives:', confmtx[0,1])
    print('false negatives:', confmtx[1,0], 
          '    true positives:', confmtx[1,1])
    print()
    unique, counts = np.unique(y_true,return_counts=True)
    
    sens =  confmtx[1,1]/counts[1]
    spec =  confmtx[0,0]/counts[0]
    print('sensitivity (TP/P):', np.round(sens*100,2), '%')  
    prec = confmtx[1,1]/(confmtx[1,1]+confmtx[0,1])
    print('precision P (TP/(TP+FP)):', np.round(prec*100,2),'%' )    
    print('specificity (TN/N):', np.round(spec*100,2), '%')      
    prec = confmtx[0,0]/(confmtx[0,0]+confmtx[1,0])
    print('precision N (TN/(TN+FN)):', np.round(prec*100,2),'%' )
    print()    
    print('accuracy:', np.round( (confmtx[1,1] + confmtx[0,0])/y_true.shape[0]*100,2),'%') # (TP + TN)/(P + N)
    
#     print()
#     print('G-mean: ', round(np.sqrt(sens*spec),2))
#     print()

#     print('MCC: ', matthews_corrcoef(y_true,y_pred))
#     print()
    
#     print('F1-measure: ',  round(fbeta_score(y_true, y_pred, beta=1.0),5))
#     print('F0.5-measure (min false positives): ',  round(fbeta_score(y_true, y_pred, beta=0.5),5))
#     print('F2-measure (min false negatives)  : ',  round(fbeta_score(y_true, y_pred, beta=2.0),5))
    print()
        
    return

# ---------------------------------
def plot_roc(rfc, test_features, test_labels):
    ax = plt.gca()
    rf_disp = RocCurveDisplay.from_estimator(rfc, test_features, test_labels, ax=ax)
    return

# ---------------------------------
def print_abs_errors(predictions_class, test_labels):# Calculate the absolute errors
    errors_class = abs(predictions_class - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors_class), 2))
    return

# ---------------------------------
def print_threshold_metrics(test_labels, predictions):
    print()
    print_accuracy_info(test_labels,predictions)
    #plot_roc(rfc, test_features, test_labels)
    print()
    return