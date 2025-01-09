# Author: Shuo Li
# Date: 2024/01/30

import os
import sys
import torch
import random
import matlab
import matlab.engine
import numpy as np
import pandas as pd
from tqdm import tqdm
from tsai.models import InceptionTime
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
import matplotlib.pyplot as plt
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data
import H_InceptionTime  # Hybrid InceptionTime model utils.
import ROCKET  # ROCKET model utils.
import MiniROCKET  # MiniROCKET model utils.
import MultiROCKET  # MultiROCKET model utils.
import HYDRA  # HYDRA model utils.

def main_Benchmark_developed(Params):
    """Main function of training a selected developed algorithm for IMU soft biometrics and report the performance.

    Parameters
    ----------
    Params: The pre-defined class of parameter settings.
    Params.method: Selected algorithm for training. 
            - ['Combined']: Manually designed features + Autocorrelation-based features + ML algorithms.
            - ['InceptionTime']: InceptionTime model.
                     References:
                     [1] Wang, Z., Yan, W., & Oates, T. (2017, May). Time series classification from scratch with deep neural networks: A strong baseline. In 2017 International joint conference on neural networks (IJCNN) (pp. 1578-1585). IEEE.
                     [2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
            - ['H-InceptionTime']: H-InceptionTime model.
                     References:
                     [1] Ismail-Fawaz, A., Devanne, M., Weber, J., & Forestier, G. (2022, December). Deep learning for time series classification using new hand-crafted convolution filters. In 2022 IEEE International Conference on Big Data (Big Data) (pp. 972-981). IEEE.
            - ['ROCKET']: ROCKET model.
                     References:
                     [1] Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery, 34(5), 1454-1495.
            - ['MiniROCKET']: MiniROCKET model.
                     References:
                     [1] Dempster, A., Schmidt, D. F., & Webb, G. I. (2021, August). Minirocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).
            - ['MultiROCKET']: MultiROCKET model.
                     References:
                     [1] Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Mining and Knowledge Discovery, 36(5), 1623-1646.
            - ['HYDRA']: HYDRA model.
                     References:
                     [1] Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery, 37(5), 1779-1805.

    Params.task: Task of age regression, age classification, or gender classification. 
                 ['age_regression', 'age_classification', 'gender_classification'].

    Returns
    -------

    """
    random.seed(Params.seed)

    if (Params.method == 'Combined'):  # Hand-crafted feature + Autocorrelation feature + ML models.
        # Set ML models.
        if (Params.task == 'age_classification') or (Params.task == 'gender_classification'):
            list_name_model = ['KM', 'DT', 'DA', 'NB', 'SVM', 'KNN', 'Ensemble']
        elif Params.task == 'age_regression':
            list_name_model = ['LR', 'DT', 'SVM', 'Ensemble', 'GPR', 'KM']
        else:
            sys.exit()
        # Loop over all ML models.
        for name_model in list_name_model:
            print(name_model)
            # Collect IMU feature data.
            list_condition = [0, 1]
            df_feature = pd.DataFrame([])
            # Loop over all walking sequences.
            for condition in list_condition:
                dir_feature_manual = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                                  'feature_manual', 'feature_manual_con_'+str(condition)+'.csv')
                df_feature_manual_tmp = pd.read_csv(dir_feature_manual)
                dir_feature_autocorr = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                                    'feature_autocorr', 'feature_autocorr_con_'+str(condition)+'.csv')
                df_feature_autocorr_tmp = pd.read_csv(dir_feature_autocorr)
                # Manually designed features + Autocorrelation-based features + ML/DL algorithms.
                df_feature_tmp = pd.concat([df_feature_manual_tmp.drop(list(set(df_feature_manual_tmp.columns) & 
                                                                            set(df_feature_autocorr_tmp.columns)), axis=1), 
                                            df_feature_autocorr_tmp], axis=1)
                df_feature = pd.concat((df_feature, df_feature_tmp), ignore_index=True)            
            # N-fold cross-validation.
            list_idx_split = util_data.split_balance(list_idx=df_feature.index.values.tolist(), 
                                                     list_age=df_feature['age'].values.tolist(), 
                                                     list_gender=df_feature['gender'].values.tolist(), 
                                                     range_age=Params.range_age, 
                                                     num_fold=Params.num_fold, 
                                                     seed=Params.seed)
            # History of training and testing.
            if Params.task == 'age_regression':
                df_feature = df_feature.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
                df_history = pd.DataFrame(columns=['fold', 'MAE_train', 'MAE_test'])
            elif Params.task == 'age_classification':
                # Create age label.
                df_feature['age_label'] = 0
                for i_age in range(0, len(Params.range_age)-1):
                    df_feature.loc[(df_feature['age'].values >= Params.range_age[i_age]) & 
                                   (df_feature['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
                df_feature = df_feature.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
                df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
            elif Params.task == 'gender_classification':
                df_feature = df_feature.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
                df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
            else:
                sys.exit()
            # Loop over each fold.
            eng = matlab.engine.start_matlab()
            for fold_tmp in tqdm(range(0, Params.num_fold)):
                # Train-test split.
                df_feature_train = pd.DataFrame(columns=df_feature.columns)
                for i_fold in range(0, len(list_idx_split)):
                    if i_fold == fold_tmp:
                        continue
                    else:
                        df_feature_train = pd.concat([df_feature_train, df_feature.loc[list_idx_split[i_fold], :]])
                df_feature_test = df_feature.loc[list_idx_split[fold_tmp], :]
                # Start training the ML model.
                if Params.task == 'age_regression':  # Age regression task.
                    # Collect data for training and testing.
                    X_train = matlab.double(df_feature_train.drop('age', axis=1).values.tolist())
                    Y_train_target = matlab.double(df_feature_train['age'].values.tolist())
                    X_test = matlab.double(df_feature_test.drop('age', axis=1).values.tolist())
                    Y_test_target = matlab.double(df_feature_test['age'].values.tolist())
                    # Fit model.
                    if name_model == 'LR':
                        model = eng.fitrlinear(X_train, Y_train_target)
                    elif name_model == 'DT':
                        model = eng.fitrtree(X_train, Y_train_target, 'MinLeafSize', 4)
                    elif name_model == 'SVM':
                        model = eng.fitrsvm(X_train, Y_train_target, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', 1)
                    elif name_model == 'Ensemble':
                        model = eng.fitrensemble(X_train, Y_train_target, 'Method', 'Bag', 'Learners', 'tree')
                    elif name_model == 'GPR':
                        model = eng.fitrgp(X_train, Y_train_target, 'KernelFunction', 'rationalquadratic', 
                                           'BasisFunction', 'constant', 'Standardize', 1, 'ConstantSigma', False)
                    else:
                        model = eng.fitrkernel(X_train, Y_train_target, 'Learner', 'leastsquares', 'KernelScale', 'auto', 'Standardize', 1)
                    
                elif (Params.task == 'age_classification') or (Params.task == 'gender_classification'):  # Classification tasks.
                    # Collect data for training and testing.
                    if Params.task == 'age_classification':
                        X_train = matlab.double(df_feature_train.drop('age_label', axis=1).values.tolist())
                        Y_train_target = matlab.double(df_feature_train['age_label'].values.tolist())
                        X_test = matlab.double(df_feature_test.drop('age_label', axis=1).values.tolist())
                        Y_test_target = matlab.double(df_feature_test['age_label'].values.tolist())
                    else:
                        X_train = matlab.double(df_feature_train.drop('gender', axis=1).values.tolist())
                        Y_train_target = matlab.double(df_feature_train['gender'].values.tolist())
                        X_test = matlab.double(df_feature_test.drop('gender', axis=1).values.tolist())
                        Y_test_target = matlab.double(df_feature_test['gender'].values.tolist())
                    # Fit model.
                    if name_model == 'DT':
                        model = eng.fitctree(X_train, Y_train_target, 'MaxNumSplits', 100)
                    elif name_model == 'DA':
                        model = eng.fitcdiscr(X_train, Y_train_target, 'DiscrimType', 'linear')
                    elif name_model == 'NB':
                        model = eng.fitcnb(X_train, Y_train_target, 'DistributionNames', 'kernel', 'Kernel', 'normal', 'Standardize', 1)
                    elif name_model == 'SVM':
                        t = eng.templateSVM('Standardize', True, 'KernelFunction', 'polynomial', 'type', 'Classification')
                        model = eng.fitcecoc(X_train, Y_train_target, 'Learners', t, 'FitPosterior', True)
                    elif name_model == 'KNN':
                        model = eng.fitcknn(X_train, Y_train_target, 'NumNeighbors', 1, 'Distance', 'euclidean', 'Standardize', 1)
                    elif name_model == 'Ensemble':
                        model = eng.fitcensemble(X_train, Y_train_target, 'Method', 'Bag', 'Learners', 'tree')
                    else:
                        t = eng.templateKernel('Learner', 'svm', 'KernelScale', 'auto', 'Standardize', 1)
                        model = eng.fitcecoc(X_train, Y_train_target, 'Learners', t)
                else:
                    sys.exit()
                # Model evaluation.
                Y_train_pred = eng.predict(model, X_train)
                Y_test_pred = eng.predict(model, X_test)
                # Record training history.
                if Params.task == 'age_regression':
                    # Matlab --> Numpy.
                    Y_train_target = np.array(Y_train_target).ravel()
                    Y_train_pred = np.array(Y_train_pred).ravel()
                    Y_test_target = np.array(Y_test_target).ravel()
                    Y_test_pred = np.array(Y_test_pred).ravel()
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 
                                                 'MAE_train': np.mean(np.abs(Y_train_target - Y_train_pred)), 
                                                 'MAE_test': np.mean(np.abs(Y_test_target - Y_test_pred)), 
                                                 'RMSE_train': np.sqrt(((Y_train_target - Y_train_pred)**2).mean()), 
                                                 'RMSE_test': np.sqrt(((Y_test_target - Y_test_pred)**2).mean()), 
                                                 'PCC_train': np.corrcoef(Y_train_target, Y_train_pred)[0, 1], 
                                                 'PCC_test': np.corrcoef(Y_test_target, Y_test_pred)[0, 1]
                                                }, index=[0])), ignore_index=True)
                elif (Params.task == 'age_classification') or (Params.task == 'gender_classification'):
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 
                                                'Acc_train': np.mean(np.array(Y_train_target).ravel() == np.array(Y_train_pred).ravel()),
                                                'Acc_test': np.mean(np.array(Y_test_target).ravel() == np.array(Y_test_pred).ravel())
                                                }, index=[0])), ignore_index=True)
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+name_model+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)

    elif (Params.method == 'InceptionTime'):  # Original IMU signal + InceptionTime model.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            dir_IMU_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                   'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU_tmp)
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_IMU.index.values.tolist(), 
                                                 list_age=df_IMU['age'].values.tolist(), 
                                                 list_gender=df_IMU['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_IMU['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_IMU.loc[(df_IMU['age'].values >= Params.range_age[i_age]) & 
                           (df_IMU['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                    if i_fold == fold_tmp:
                        continue
                    else:
                        df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro_x = df_IMU_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_y = df_IMU_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_z = df_IMU_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_train_acc_x = df_IMU_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_y = df_IMU_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_z = df_IMU_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train = np.concatenate((data_train_gyro_x[:, np.newaxis, :], 
                                         data_train_gyro_y[:, np.newaxis, :], 
                                         data_train_gyro_z[:, np.newaxis, :], 
                                         data_train_acc_x[:, np.newaxis, :], 
                                         data_train_acc_y[:, np.newaxis, :], 
                                         data_train_acc_z[:, np.newaxis, :]), axis=1)
            # Test set.
            df_IMU_test = df_IMU.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro_x = df_IMU_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_y = df_IMU_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_z = df_IMU_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_test_acc_x = df_IMU_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_y = df_IMU_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_z = df_IMU_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test = np.concatenate((data_test_gyro_x[:, np.newaxis, :], 
                                        data_test_gyro_y[:, np.newaxis, :], 
                                        data_test_gyro_z[:, np.newaxis, :], 
                                        data_test_acc_x[:, np.newaxis, :], 
                                        data_test_acc_y[:, np.newaxis, :], 
                                        data_test_acc_z[:, np.newaxis, :]), axis=1)
            # Move tensors to the target device.
            data_test = torch.tensor(data_test).to(Params.device)
            
            if Params.task == 'age_regression':  # Age regression.
                # Load model.
                model = InceptionTime.InceptionTime(c_in=6, c_out=1, seq_len=300, nf=32, nb_filters=None, ks=40, bottleneck=True)
                age_train = df_IMU_train.loc[:, ['age']].values.squeeze()  # Training set.
                age_test = df_IMU_test.loc[:, ['age']].values.squeeze()  # Test set.
                age_test = torch.tensor(age_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.HuberLoss()
            elif Params.task == 'age_classification':  # Age group classification.
                # Load model.
                model = InceptionTime.InceptionTime(c_in=6, c_out=len(Params.range_age)+1, seq_len=300, nf=32, nb_filters=None, ks=40, bottleneck=True)
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.squeeze()  # Training set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.squeeze()  # Test set.
                age_label_test = torch.tensor(age_label_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            elif Params.task == 'gender_classification':
                # Load model.
                model = InceptionTime.InceptionTime(c_in=6, c_out=2, seq_len=300, nf=32, nb_filters=None, ks=40, bottleneck=True)
                gender_train = df_IMU_train.loc[:, ['gender']].values.squeeze()  # Training set.
                gender_test = df_IMU_test.loc[:, ['gender']].values.squeeze()  # Test set.
                gender_test = torch.tensor(gender_test, dtype=torch.float32).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            else:
                # Other conditions. Stop training.
                sys.exit()
            model.train()
            model = model.to(Params.device)
            # Optimizer.
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            # Start training.
            epochs = 1000
            for epoch in tqdm(range(epochs)):
                # Zero gradient.
                optimizer.zero_grad()
                # Select a subset from the training data.
                list_idx = np.random.choice(range(0, len(df_IMU_train)), 300)
                data_train_tmp = data_train[list_idx]  # Training data.
                data_train_tmp = torch.tensor(data_train_tmp).to(Params.device)  # Move the training data to target device.

                if Params.task == 'age_regression':
                    # Target age (ground truth).
                    age_train_tmp = age_train[list_idx].astype(dtype=np.float32)
                    age_train_tmp = torch.tensor(age_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(age_pred_tmp, age_train_tmp.reshape_as(age_pred_tmp).float())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training MAE.
                    mae_train = torch.abs(age_pred_tmp.ravel() - age_train_tmp.ravel()).mean()
                    # Compute training RMSE.
                    rmse_train = torch.sqrt(((age_pred_tmp.ravel() - age_train_tmp.ravel())**2).mean())
                    # Compute training PCC.
                    pcc_train = np.corrcoef(age_pred_tmp.cpu().detach().numpy().ravel(), age_train_tmp.cpu().detach().numpy().ravel())[0, 1]
                    # Compute testing loss.
                    age_pred_tmp = model(data_test.type(torch.float32))
                    loss_test = criterion(age_pred_tmp, age_test.reshape_as(age_pred_tmp).float())
                    # Compute testing MAE.
                    mae_test = torch.abs((age_pred_tmp.ravel() - age_test.ravel())).mean()
                    # Compute testing RMSE.
                    rmse_test = torch.sqrt(((age_pred_tmp.ravel() - age_test.ravel())**2).mean())
                    # Compute testing PCC.
                    pcc_test = np.corrcoef(age_pred_tmp.cpu().detach().numpy().ravel(), age_test.cpu().detach().numpy().ravel())[0, 1]
                    # Record training history.
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 'epoch': epoch, 
                                                 'Loss_train': loss_train.item(), 'Loss_test': loss_test.item(), 
                                                 'MAE_train': mae_train.item(), 'MAE_test': mae_test.item(), 
                                                 'RMSE_train': rmse_train.item(), 'RMSE_test': rmse_test.item(), 
                                                 'PCC_train': pcc_train.item(), 'PCC_test': pcc_test.item(), 
                                                 }, index=[0])), ignore_index=True)
                elif Params.task == 'age_classification':
                    # Target age class (ground truth).
                    age_label_train_tmp = age_label_train[list_idx].astype(np.float32)
                    age_label_train_tmp = torch.tensor(age_label_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(age_pred_tmp, age_label_train_tmp.long())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training accuracy.
                    # Predicted age classes of the training data.
                    age_label_pred_tmp = torch.zeros(size=[age_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(age_label_pred_tmp.shape[0]):
                        age_label_pred_tmp[i_1] = torch.argmax(age_pred_tmp[i_1])
                    acc_train = ((age_label_pred_tmp == age_label_train_tmp).sum())/age_label_pred_tmp.shape[0]
                    # Compute testing loss.
                    age_pred_tmp = model(data_test.type(torch.float32))
                    loss_test = criterion(age_pred_tmp, age_label_test.long())
                    # Compute testing accuracy.
                    age_label_pred_tmp = torch.zeros(size=[age_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(age_label_pred_tmp.shape[0]):
                        age_label_pred_tmp[i_1] = torch.argmax(age_pred_tmp[i_1])
                    acc_test = ((age_label_pred_tmp == age_label_test).sum())/age_label_pred_tmp.shape[0]
                    # Record training history.
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 'epoch': epoch, 
                                                 'Loss_train': loss_train.item(), 'Loss_test': loss_test.item(), 
                                                 'Acc_train': acc_train.item(), 'Acc_test': acc_test.item()
                                                 }, index=[0])), ignore_index=True)
                elif Params.task == 'gender_classification':
                    # Target gender (ground truth).
                    gender_train_tmp = gender_train[list_idx].astype(np.int32)
                    gender_train_tmp = torch.tensor(gender_train_tmp).to(Params.device)
                    # Predicted gender.
                    gender_pred_tmp = model(data_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(gender_pred_tmp, gender_train_tmp.long())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training accuracy.
                    gender_gt_tmp = torch.zeros(size=[gender_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(gender_pred_tmp.shape[0]):
                        if gender_pred_tmp[i_1, 0] >= gender_pred_tmp[i_1, 1]:
                            gender_gt_tmp[i_1] = 0
                        else:
                            gender_gt_tmp[i_1] = 1
                    acc_train = ((gender_gt_tmp == gender_train_tmp).sum())/gender_pred_tmp.shape[0]
                    # Compute testing loss.
                    gender_pred_tmp = model(data_test.type(torch.float32))
                    loss_test = criterion(gender_pred_tmp, gender_test.long())
                    # Compute testing accuracy.
                    gender_gt_tmp = torch.zeros(size=[gender_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(gender_pred_tmp.shape[0]):
                        if gender_pred_tmp[i_1, 0] >= gender_pred_tmp[i_1, 1]:
                            gender_gt_tmp[i_1] = 0
                        else:
                            gender_gt_tmp[i_1] = 1
                    acc_test = ((gender_gt_tmp == gender_test).sum())/gender_pred_tmp.shape[0]
                    # Record training history.
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 'epoch': epoch, 
                                                 'Loss_train': loss_train.item(), 'Loss_test': loss_test.item(), 
                                                 'Acc_train': acc_train.item(), 'Acc_test': acc_test.item()
                                                 }, index=[0])), ignore_index=True)
                else:
                    sys.exit()
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)

    elif (Params.method == 'H-InceptionTime'):  # Original IMU signal + H-InceptionTime model.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            dir_IMU_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                   'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU_tmp)
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_IMU.index.values.tolist(), 
                                                 list_age=df_IMU['age'].values.tolist(), 
                                                 list_gender=df_IMU['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_IMU['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_IMU.loc[(df_IMU['age'].values >= Params.range_age[i_age]) & 
                           (df_IMU['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                    if i_fold == fold_tmp:
                        continue
                    else:
                        df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro_x = df_IMU_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_y = df_IMU_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_z = df_IMU_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_train_acc_x = df_IMU_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_y = df_IMU_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_z = df_IMU_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train = np.concatenate((data_train_gyro_x[:, np.newaxis, :], 
                                         data_train_gyro_y[:, np.newaxis, :], 
                                         data_train_gyro_z[:, np.newaxis, :], 
                                         data_train_acc_x[:, np.newaxis, :], 
                                         data_train_acc_y[:, np.newaxis, :], 
                                         data_train_acc_z[:, np.newaxis, :]), axis=1)
            # Test set.
            df_IMU_test = df_IMU.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro_x = df_IMU_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_y = df_IMU_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_z = df_IMU_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_test_acc_x = df_IMU_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_y = df_IMU_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_z = df_IMU_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test = np.concatenate((data_test_gyro_x[:, np.newaxis, :], 
                                        data_test_gyro_y[:, np.newaxis, :], 
                                        data_test_gyro_z[:, np.newaxis, :], 
                                        data_test_acc_x[:, np.newaxis, :], 
                                        data_test_acc_y[:, np.newaxis, :], 
                                        data_test_acc_z[:, np.newaxis, :]), axis=1)
            # Move tensors to the target device.
            data_test = torch.tensor(data_test).to(Params.device)
            
            if Params.task == 'age_regression':  # Age regression.
                # Load model.
                model = H_InceptionTime.HybridInceptionTime(in_channels=6, number_classes=1)
                age_train = df_IMU_train.loc[:, ['age']].values.squeeze()  # Training set.
                age_test = df_IMU_test.loc[:, ['age']].values.squeeze()  # Test set.
                age_test = torch.tensor(age_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.HuberLoss()
            elif Params.task == 'age_classification':  # Age group classification.
                # Load model.
                model = H_InceptionTime.HybridInceptionTime(in_channels=6, number_classes=len(Params.range_age)+1)
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.squeeze()  # Training set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.squeeze()  # Test set.
                age_label_test = torch.tensor(age_label_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            elif Params.task == 'gender_classification':
                # Load model.
                model = H_InceptionTime.HybridInceptionTime(in_channels=6, number_classes=2)
                gender_train = df_IMU_train.loc[:, ['gender']].values.squeeze()  # Training set.
                gender_test = df_IMU_test.loc[:, ['gender']].values.squeeze()  # Test set.
                gender_test = torch.tensor(gender_test, dtype=torch.float32).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            else:
                # Other conditions. Stop training.
                sys.exit()
            model.train()
            model = model.to(Params.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            # Start training.
            epochs = 1000
            for epoch in tqdm(range(epochs)):
                # Zero gradient.
                optimizer.zero_grad()
                # Select a subset from the training data.
                list_idx = np.random.choice(range(0, len(df_IMU_train)), 300)
                data_train_tmp = data_train[list_idx]  # Training data.
                data_train_tmp = torch.tensor(data_train_tmp).to(Params.device)  # Move the training data to target device.

                if Params.task == 'age_regression':
                    # Target age (ground truth).
                    age_train_tmp = age_train[list_idx].astype(dtype=np.float32)
                    age_train_tmp = torch.tensor(age_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(age_pred_tmp, age_train_tmp.reshape_as(age_pred_tmp).float())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training MAE.
                    mae_train = torch.abs((age_pred_tmp.ravel() - age_train_tmp.ravel())).mean()
                    # Compute training RMSE.
                    rmse_train = torch.sqrt(((age_pred_tmp.ravel() - age_train_tmp.ravel())**2).mean())
                    # Compute training PCC.
                    pcc_train = np.corrcoef(age_pred_tmp.cpu().detach().numpy().ravel(), age_train_tmp.cpu().detach().numpy().ravel())[0, 1]
                    # Compute testing loss.
                    age_pred_tmp = model(data_test.type(torch.float32))
                    loss_test = criterion(age_pred_tmp, age_test.reshape_as(age_pred_tmp).float())
                    # Compute testing MAE.
                    mae_test = torch.abs((age_pred_tmp.ravel() - age_test.ravel())).mean()
                    # Compute testing RMSE.
                    rmse_test = torch.sqrt(((age_pred_tmp.ravel() - age_test.ravel())**2).mean())
                    # Compute testing PCC.
                    pcc_test = np.corrcoef(age_pred_tmp.cpu().detach().numpy().ravel(), age_test.cpu().detach().numpy().ravel())[0, 1]
                    # Record training history.
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 'epoch': epoch, 
                                                 'Loss_train': loss_train.item(), 'Loss_test': loss_test.item(), 
                                                 'MAE_train': mae_train.item(), 'MAE_test': mae_test.item(), 
                                                 'RMSE_train': rmse_train.item(), 'RMSE_test': rmse_test.item(), 
                                                 'PCC_train': pcc_train.item(), 'PCC_test': pcc_test.item(), 
                                                 }, index=[0])), ignore_index=True)
                elif Params.task == 'age_classification':
                    # Target age class (ground truth).
                    age_label_train_tmp = age_label_train[list_idx].astype(np.float32)
                    age_label_train_tmp = torch.tensor(age_label_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(age_pred_tmp, age_label_train_tmp.long())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training accuracy.
                    # Predicted age classes of the training data.
                    age_label_pred_tmp = torch.zeros(size=[age_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(age_label_pred_tmp.shape[0]):
                        age_label_pred_tmp[i_1] = torch.argmax(age_pred_tmp[i_1])
                    acc_train = ((age_label_pred_tmp == age_label_train_tmp).sum())/age_label_pred_tmp.shape[0]
                    # Compute testing loss.
                    age_pred_tmp = model(data_test.type(torch.float32))
                    loss_test = criterion(age_pred_tmp, age_label_test.long())
                    # Compute testing accuracy.
                    age_label_pred_tmp = torch.zeros(size=[age_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(age_label_pred_tmp.shape[0]):
                        age_label_pred_tmp[i_1] = torch.argmax(age_pred_tmp[i_1])
                    acc_test = ((age_label_pred_tmp == age_label_test).sum())/age_label_pred_tmp.shape[0]
                    # Record training history.
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 'epoch': epoch, 
                                                 'Loss_train': loss_train.item(), 'Loss_test': loss_test.item(), 
                                                 'Acc_train': acc_train.item(), 'Acc_test': acc_test.item()
                                                 }, index=[0])), ignore_index=True)
                elif Params.task == 'gender_classification':
                    # Target gender (ground truth).
                    gender_train_tmp = gender_train[list_idx].astype(np.int32)
                    gender_train_tmp = torch.tensor(gender_train_tmp).to(Params.device)
                    # Predicted gender.
                    gender_pred_tmp = model(data_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(gender_pred_tmp, gender_train_tmp.long())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training accuracy.
                    gender_gt_tmp = torch.zeros(size=[gender_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(gender_pred_tmp.shape[0]):
                        if gender_pred_tmp[i_1, 0] >= gender_pred_tmp[i_1, 1]:
                            gender_gt_tmp[i_1] = 0
                        else:
                            gender_gt_tmp[i_1] = 1
                    acc_train = ((gender_gt_tmp == gender_train_tmp).sum())/gender_pred_tmp.shape[0]
                    # Compute testing loss.
                    gender_pred_tmp = model(data_test.type(torch.float32))
                    loss_test = criterion(gender_pred_tmp, gender_test.long())
                    # Compute testing accuracy.
                    gender_gt_tmp = torch.zeros(size=[gender_pred_tmp.shape[0]], device=Params.device)
                    for i_1 in range(gender_pred_tmp.shape[0]):
                        if gender_pred_tmp[i_1, 0] >= gender_pred_tmp[i_1, 1]:
                            gender_gt_tmp[i_1] = 0
                        else:
                            gender_gt_tmp[i_1] = 1
                    acc_test = ((gender_gt_tmp == gender_test).sum())/gender_pred_tmp.shape[0]
                    # Record training history.
                    df_history = pd.concat((df_history, 
                                            pd.DataFrame(
                                                {'fold': fold_tmp, 'epoch': epoch, 
                                                 'Loss_train': loss_train.item(), 'Loss_test': loss_test.item(), 
                                                 'Acc_train': acc_train.item(), 'Acc_test': acc_test.item()
                                                 }, index=[0])), ignore_index=True)
                else:
                    sys.exit()
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)

    elif Params.method == 'ROCKET':  # ROCKET model.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            dir_IMU_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                       'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU_tmp)
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_IMU.index.values.tolist(), 
                                                 list_age=df_IMU['age'].values.tolist(), 
                                                 list_gender=df_IMU['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_IMU['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_IMU.loc[(df_IMU['age'].values >= Params.range_age[i_age]) & 
                           (df_IMU['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                if i_fold == fold_tmp:
                    continue
                else:
                    df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro_x = df_IMU_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_y = df_IMU_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_z = df_IMU_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_train_acc_x = df_IMU_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_y = df_IMU_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_z = df_IMU_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_ori = np.concatenate((data_train_gyro_x[:, np.newaxis, :], 
                                             data_train_gyro_y[:, np.newaxis, :], 
                                             data_train_gyro_z[:, np.newaxis, :], 
                                             data_train_acc_x[:, np.newaxis, :], 
                                             data_train_acc_y[:, np.newaxis, :], 
                                             data_train_acc_z[:, np.newaxis, :]), axis=1)
            # Test set.
            df_IMU_test = df_IMU.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro_x = df_IMU_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_y = df_IMU_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_z = df_IMU_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_test_acc_x = df_IMU_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_y = df_IMU_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_z = df_IMU_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_ori = np.concatenate((data_test_gyro_x[:, np.newaxis, :], 
                                            data_test_gyro_y[:, np.newaxis, :], 
                                            data_test_gyro_z[:, np.newaxis, :], 
                                            data_test_acc_x[:, np.newaxis, :], 
                                            data_test_acc_y[:, np.newaxis, :], 
                                            data_test_acc_z[:, np.newaxis, :]), axis=1)
            # Transform into tensors.
            data_train = torch.Tensor([])
            data_test = torch.Tensor([])
            # Transform the original data into ROCKET features.
            input_length = data_train_ori.shape[-1]
            kernels = ROCKET.generate_kernels(input_length, 10_000)
            for i_channel in range(data_train_ori.shape[1]):
                data_train = torch.cat((data_train, 
                                        torch.from_numpy(ROCKET.apply_kernels(data_train_ori[:, i_channel, :], kernels))), 1)
                data_test = torch.cat((data_test, 
                                       torch.from_numpy(ROCKET.apply_kernels(data_test_ori[:, i_channel, :], kernels))), 1)
            # Fit and evaluate models.
            if Params.task == 'age_regression':  # Age regression.
                # Age target.
                # Training set.
                age_train = df_IMU_train.loc[:, ['age']].values.ravel().astype(dtype=np.float32)
                age_train = torch.tensor(age_train, dtype=torch.float32)
                # Test set.
                age_test = df_IMU_test.loc[:, ['age']].values.ravel()
                age_test = torch.tensor(age_test, dtype=torch.float32)
                # Ridge regressor.
                model = RidgeCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_train)
                age_train_pred = model.predict(data_train)
                mae_train = np.abs(age_train_pred - age_train.numpy()).mean()   # MAE train.
                rmse_train = np.sqrt(np.mean((age_train_pred - age_train.numpy())**2))   # RMSE train.
                pcc_train = np.corrcoef(age_train_pred, age_train)[0, 1]   # PCC train.
                age_test_pred = model.predict(data_test)
                mae_test = np.abs(age_test_pred - age_test.numpy()).mean()   # MAE test.
                rmse_test = np.sqrt(np.mean((age_test_pred - age_test.numpy())**2))   # RMSE test.
                pcc_test = np.corrcoef(age_test_pred, age_test)[0, 1]   # PCC test.
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'MAE_train': mae_train.item(), 
                                             'MAE_test': mae_test.item(), 
                                             'RMSE_train': rmse_train.item(), 
                                             'RMSE_test': rmse_test.item(), 
                                             'PCC_train': pcc_train.item(), 
                                             'PCC_test': pcc_test.item(),
                                             }, index=[0])), ignore_index=True)
            elif Params.task == 'age_classification':  # Age group classification.
                # Age class target.
                # Training set.
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.ravel().astype(np.float32)
                age_label_train = torch.tensor(age_label_train, dtype=torch.float32)
                # Test set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.ravel()
                age_label_test = torch.tensor(age_label_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_label_train)
                acc_train = model.score(data_train, age_label_train)
                acc_test = model.score(data_test, age_label_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            elif Params.task == 'gender_classification':  # Gender classification.
                # Gender class target.
                # Training set.
                gender_train = df_IMU_train.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_train = torch.tensor(gender_train, dtype=torch.float32)
                # Test set.
                gender_test = df_IMU_test.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_test = torch.tensor(gender_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, gender_train)
                acc_train = model.score(data_train, gender_train)
                acc_test = model.score(data_test, gender_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            else:
                # Other conditions. Stop training.
                sys.exit()
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)

    elif Params.method == 'MiniROCKET':  # MiniROCKET model.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            dir_IMU_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                       'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU_tmp)
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_IMU.index.values.tolist(), 
                                                 list_age=df_IMU['age'].values.tolist(), 
                                                 list_gender=df_IMU['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_IMU['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_IMU.loc[(df_IMU['age'].values >= Params.range_age[i_age]) & 
                           (df_IMU['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                if i_fold == fold_tmp:
                    continue
                else:
                    df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro_x = df_IMU_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_y = df_IMU_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_z = df_IMU_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_train_acc_x = df_IMU_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_y = df_IMU_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_z = df_IMU_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_ori = np.concatenate((data_train_gyro_x[:, np.newaxis, :], 
                                             data_train_gyro_y[:, np.newaxis, :], 
                                             data_train_gyro_z[:, np.newaxis, :], 
                                             data_train_acc_x[:, np.newaxis, :], 
                                             data_train_acc_y[:, np.newaxis, :], 
                                             data_train_acc_z[:, np.newaxis, :]), axis=1)
            # Test set.
            df_IMU_test = df_IMU.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro_x = df_IMU_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_y = df_IMU_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_z = df_IMU_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_test_acc_x = df_IMU_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_y = df_IMU_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_z = df_IMU_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_ori = np.concatenate((data_test_gyro_x[:, np.newaxis, :], 
                                            data_test_gyro_y[:, np.newaxis, :], 
                                            data_test_gyro_z[:, np.newaxis, :], 
                                            data_test_acc_x[:, np.newaxis, :], 
                                            data_test_acc_y[:, np.newaxis, :], 
                                            data_test_acc_z[:, np.newaxis, :]), axis=1)
            # Transform into tensors.
            data_train_ori = data_train_ori.astype(np.float32)
            data_test_ori = data_test_ori.astype(np.float32)
            data_train = torch.Tensor([])
            data_test = torch.Tensor([])
            # Transform the original data into MiniROCKET features.
            for i_channel in range(data_train_ori.shape[1]):
                parameter_fit = MiniROCKET.fit(data_train_ori[:, i_channel, :], num_features=20_000, max_dilations_per_kernel=32)
                data_train = torch.cat((data_train, 
                                        torch.from_numpy(MiniROCKET.transform(data_train_ori[:, i_channel, :], parameter_fit))), 1)
                data_test = torch.cat((data_test, 
                                       torch.from_numpy(MiniROCKET.transform(data_test_ori[:, i_channel, :], parameter_fit))), 1)
            # Fit and evaluate models.
            if Params.task == 'age_regression':  # Age regression.
                # Age target.
                # Training set.
                age_train = df_IMU_train.loc[:, ['age']].values.ravel().astype(dtype=np.float32)
                age_train = torch.tensor(age_train, dtype=torch.float32)
                # Test set.
                age_test = df_IMU_test.loc[:, ['age']].values.ravel()
                age_test = torch.tensor(age_test, dtype=torch.float32)
                # Ridge regressor.
                model = RidgeCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_train)

                age_train_pred = model.predict(data_train)
                mae_train = np.abs(age_train_pred - age_train.numpy()).mean()   # MAE train.
                rmse_train = np.sqrt(np.mean((age_train_pred - age_train.numpy())**2))   # RMSE train.
                pcc_train = np.corrcoef(age_train_pred, age_train)[0, 1]   # PCC train.
                age_test_pred = model.predict(data_test)
                mae_test = np.abs(age_test_pred - age_test.numpy()).mean()   # MAE test.
                rmse_test = np.sqrt(np.mean((age_test_pred - age_test.numpy())**2))   # RMSE test.
                pcc_test = np.corrcoef(age_test_pred, age_test)[0, 1]   # PCC test.
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'MAE_train': mae_train.item(), 
                                             'MAE_test': mae_test.item(), 
                                             'RMSE_train': rmse_train.item(), 
                                             'RMSE_test': rmse_test.item(), 
                                             'PCC_train': pcc_train.item(), 
                                             'PCC_test': pcc_test.item(),
                                             }, index=[0])), ignore_index=True)
                
            elif Params.task == 'age_classification':  # Age group classification.
                # Age class target.
                # Training set.
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.ravel().astype(np.float32)
                age_label_train = torch.tensor(age_label_train, dtype=torch.float32)
                # Test set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.ravel()
                age_label_test = torch.tensor(age_label_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_label_train)
                acc_train = model.score(data_train, age_label_train)
                acc_test = model.score(data_test, age_label_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            elif Params.task == 'gender_classification':  # Gender classification.
                # Gender class target.
                # Training set.
                gender_train = df_IMU_train.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_train = torch.tensor(gender_train, dtype=torch.float32)
                # Test set.
                gender_test = df_IMU_test.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_test = torch.tensor(gender_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, gender_train)
                acc_train = model.score(data_train, gender_train)
                acc_test = model.score(data_test, gender_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            else:
                # Other conditions. Stop training.
                sys.exit()
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)

    elif Params.method == 'MultiROCKET':  # MultiROCKET model.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            dir_IMU_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                       'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU_tmp)
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_IMU.index.values.tolist(), 
                                                 list_age=df_IMU['age'].values.tolist(), 
                                                 list_gender=df_IMU['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_IMU['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_IMU.loc[(df_IMU['age'].values >= Params.range_age[i_age]) & 
                           (df_IMU['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                if i_fold == fold_tmp:
                    continue
                else:
                    df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro_x = df_IMU_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_y = df_IMU_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_z = df_IMU_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_train_acc_x = df_IMU_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_y = df_IMU_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_z = df_IMU_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_ori = np.concatenate((data_train_gyro_x[:, np.newaxis, :], 
                                             data_train_gyro_y[:, np.newaxis, :], 
                                             data_train_gyro_z[:, np.newaxis, :], 
                                             data_train_acc_x[:, np.newaxis, :], 
                                             data_train_acc_y[:, np.newaxis, :], 
                                             data_train_acc_z[:, np.newaxis, :]), axis=1)
            # Test set.
            df_IMU_test = df_IMU.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro_x = df_IMU_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_y = df_IMU_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_z = df_IMU_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_test_acc_x = df_IMU_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_y = df_IMU_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_z = df_IMU_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_ori = np.concatenate((data_test_gyro_x[:, np.newaxis, :], 
                                            data_test_gyro_y[:, np.newaxis, :], 
                                            data_test_gyro_z[:, np.newaxis, :], 
                                            data_test_acc_x[:, np.newaxis, :], 
                                            data_test_acc_y[:, np.newaxis, :], 
                                            data_test_acc_z[:, np.newaxis, :]), axis=1)
            # Transform into tensors.
            data_train = torch.Tensor([])
            data_test = torch.Tensor([])
            # Transform the original data into MultiROCKET features.
            for i_channel in range(data_train_ori.shape[1]):
                data_train_ori_diff = np.diff(data_train_ori[:, i_channel, :], 1)
                data_test_ori_diff = np.diff(data_test_ori[:, i_channel, :], 1)
                parameter_base = MultiROCKET.fit(data_train_ori[:, i_channel, :], num_features=2500, max_dilations_per_kernel=32)
                parameter_diff = MultiROCKET.fit(data_train_ori_diff, num_features=2500, max_dilations_per_kernel=32)
                data_train = torch.cat((data_train, 
                                        torch.from_numpy(MultiROCKET.transform(data_train_ori[:, i_channel, :], data_train_ori_diff, 
                                                                               parameter_base, parameter_diff, 4))), 1)
                data_test = torch.cat((data_test, 
                                       torch.from_numpy(MultiROCKET.transform(data_test_ori[:, i_channel, :], data_test_ori_diff, 
                                                                              parameter_base, parameter_diff, 4))), 1)
            # Fit and evaluate models.
            if Params.task == 'age_regression':  # Age regression.
                # Age target.
                # Training set.
                age_train = df_IMU_train.loc[:, ['age']].values.ravel().astype(dtype=np.float32)
                age_train = torch.tensor(age_train, dtype=torch.float32)
                # Test set.
                age_test = df_IMU_test.loc[:, ['age']].values.ravel()
                age_test = torch.tensor(age_test, dtype=torch.float32)
                # Ridge regressor.
                model = RidgeCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_train)
                age_train_pred = model.predict(data_train)
                mae_train = np.abs(age_train_pred - age_train.numpy()).mean()   # MAE train.
                rmse_train = np.sqrt(np.mean((age_train_pred - age_train.numpy())**2))   # RMSE train.
                pcc_train = np.corrcoef(age_train_pred, age_train)[0, 1]   # PCC train.
                age_test_pred = model.predict(data_test)
                mae_test = np.abs(age_test_pred - age_test.numpy()).mean()   # MAE test.
                rmse_test = np.sqrt(np.mean((age_test_pred - age_test.numpy())**2))   # RMSE test.
                pcc_test = np.corrcoef(age_test_pred, age_test)[0, 1]   # PCC test.
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'MAE_train': mae_train.item(), 
                                             'MAE_test': mae_test.item(), 
                                             'RMSE_train': rmse_train.item(), 
                                             'RMSE_test': rmse_test.item(), 
                                             'PCC_train': pcc_train.item(), 
                                             'PCC_test': pcc_test.item(),
                                             }, index=[0])), ignore_index=True)

            elif Params.task == 'age_classification':  # Age group classification.
                # Age class target.
                # Training set.
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.ravel().astype(np.float32)
                age_label_train = torch.tensor(age_label_train, dtype=torch.float32)
                # Test set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.ravel()
                age_label_test = torch.tensor(age_label_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_label_train)
                acc_train = model.score(data_train, age_label_train)
                acc_test = model.score(data_test, age_label_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            elif Params.task == 'gender_classification':  # Gender classification.
                # Gender class target.
                # Training set.
                gender_train = df_IMU_train.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_train = torch.tensor(gender_train, dtype=torch.float32)
                # Test set.
                gender_test = df_IMU_test.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_test = torch.tensor(gender_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, gender_train)
                acc_train = model.score(data_train, gender_train)
                acc_test = model.score(data_test, gender_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            else:
                # Other conditions. Stop training.
                sys.exit()
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)

    elif Params.method == 'HYDRA':  # HYDRA model.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            dir_IMU_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                       'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU_tmp)
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_IMU.index.values.tolist(), 
                                                 list_age=df_IMU['age'].values.tolist(), 
                                                 list_gender=df_IMU['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_IMU['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_IMU.loc[(df_IMU['age'].values >= Params.range_age[i_age]) & 
                           (df_IMU['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_IMU = df_IMU.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Initialize the Hydra transform correspondance.
        transform = HYDRA.Hydra(Params.len_segment)
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                if i_fold == fold_tmp:
                    continue
                else:
                    df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro_x = df_IMU_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_y = df_IMU_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_gyro_z = df_IMU_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_train_acc_x = df_IMU_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_y = df_IMU_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_acc_z = df_IMU_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_train_ori = np.concatenate((data_train_gyro_x[:, np.newaxis, :], 
                                             data_train_gyro_y[:, np.newaxis, :], 
                                             data_train_gyro_z[:, np.newaxis, :], 
                                             data_train_acc_x[:, np.newaxis, :], 
                                             data_train_acc_y[:, np.newaxis, :], 
                                             data_train_acc_z[:, np.newaxis, :]), axis=1)
            # Test set.
            df_IMU_test = df_IMU.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro_x = df_IMU_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_y = df_IMU_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_gyro_z = df_IMU_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_test_acc_x = df_IMU_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_y = df_IMU_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_acc_z = df_IMU_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_test_ori = np.concatenate((data_test_gyro_x[:, np.newaxis, :], 
                                            data_test_gyro_y[:, np.newaxis, :], 
                                            data_test_gyro_z[:, np.newaxis, :], 
                                            data_test_acc_x[:, np.newaxis, :], 
                                            data_test_acc_y[:, np.newaxis, :], 
                                            data_test_acc_z[:, np.newaxis, :]), axis=1)
            # Transform into tensors.
            data_train_ori = torch.tensor(data_train_ori, dtype=torch.float32).unsqueeze(2)
            data_train = torch.Tensor([])
            data_test_ori = torch.tensor(data_test_ori, dtype=torch.float32).unsqueeze(2)
            data_test = torch.Tensor([])
            # Transform the original data into Hydra features.
            for i_channel in range(data_train_ori.shape[1]):
                scaler = HYDRA.SparseScaler()
                data_train = torch.cat((data_train, scaler.fit_transform(transform(data_train_ori[:, i_channel, :]))), 1)
                data_test = torch.cat((data_test, scaler.transform(transform(data_test_ori[:, i_channel, :]))), 1)
            # Fit and evaluate models.
            if Params.task == 'age_regression':  # Age regression.
                # Age target.
                # Training set.
                age_train = df_IMU_train.loc[:, ['age']].values.ravel().astype(dtype=np.float32)
                age_train = torch.tensor(age_train, dtype=torch.float32)
                # Test set.
                age_test = df_IMU_test.loc[:, ['age']].values.ravel()
                age_test = torch.tensor(age_test, dtype=torch.float32)
                # Ridge regressor.
                model = RidgeCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_train)                
                age_train_pred = model.predict(data_train)
                mae_train = np.abs(age_train_pred - age_train.numpy()).mean()   # MAE train.
                rmse_train = np.sqrt(np.mean((age_train_pred - age_train.numpy())**2))   # RMSE train.
                pcc_train = np.corrcoef(age_train_pred, age_train)[0, 1]   # PCC train.
                age_test_pred = model.predict(data_test)
                mae_test = np.abs(age_test_pred - age_test.numpy()).mean()   # MAE test.
                rmse_test = np.sqrt(np.mean((age_test_pred - age_test.numpy())**2))   # RMSE test.
                pcc_test = np.corrcoef(age_test_pred, age_test)[0, 1]   # PCC test.
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'MAE_train': mae_train.item(), 
                                             'MAE_test': mae_test.item(), 
                                             'RMSE_train': rmse_train.item(), 
                                             'RMSE_test': rmse_test.item(), 
                                             'PCC_train': pcc_train.item(), 
                                             'PCC_test': pcc_test.item(),
                                             }, index=[0])), ignore_index=True)
            elif Params.task == 'age_classification':  # Age group classification.
                # Age class target.
                # Training set.
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.ravel().astype(np.float32)
                age_label_train = torch.tensor(age_label_train, dtype=torch.float32)
                # Test set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.ravel()
                age_label_test = torch.tensor(age_label_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, age_label_train)
                acc_train = model.score(data_train, age_label_train)
                acc_test = model.score(data_test, age_label_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            elif Params.task == 'gender_classification':  # Gender classification.
                # Gender class target.
                # Training set.
                gender_train = df_IMU_train.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_train = torch.tensor(gender_train, dtype=torch.float32)
                # Test set.
                gender_test = df_IMU_test.loc[:, ['gender']].values.ravel().astype(np.float32)
                gender_test = torch.tensor(gender_test, dtype=torch.float32)
                # Ridge classifier.
                model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                model.fit(data_train, gender_train)
                acc_train = model.score(data_train, gender_train)
                acc_test = model.score(data_test, gender_test)
                # Record training history.
                df_history = pd.concat((df_history, 
                                        pd.DataFrame(
                                            {'fold': fold_tmp, 
                                             'Acc_train': acc_train, 
                                             'Acc_test': acc_test
                                             }, index=[0])), ignore_index=True)
            else:
                # Other conditions. Stop training.
                sys.exit()
            # Save the training history for each fold.
            dir_history = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_'+Params.task+'.csv')
            df_history.to_csv(dir_history, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')   # Load pre-defiend options.
    name_dataset = 'OU_ISIR_Inertial_Sensor'   # ['OU_ISIR_Inertial_Sensor'].
    Params = util_data.Params(dir_option, name_dataset)   # Initialize parameter object.
    list_method = ['Combined', 'InceptionTime', 'H-InceptionTime', 'ROCKET', 'MiniROCKET', 'MultiROCKET', 'HYDRA']
    list_type_imu = ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    list_task = ['age_regression', 'age_classification', 'gender_classification']
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        for task in list_task:
            Params.task = task
            for method in list_method:
                Params.method = method
                # Print current model information.
                print('Method: '+method+'. Data Type: '+type_imu+'. Task: '+task+'.')
                main_Benchmark_developed(Params=Params)