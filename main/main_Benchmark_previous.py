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
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data
import util_benchmark


def main_Benchmark_previous(Params):
    """Main function of training a selected previous algorithm for IMU soft biometrics and report the performance.

    Parameters
    ----------
    Params: The pre-defined class of parameter settings.
    Params.method: Selected algorithm for training. 
            - ['Khabir']:
                References: 
                [1] Khabir, K. M., Siraj, M. S., Ahmed, M., & Ahmed, M. U. (2019, May). Prediction of gender and age from inertial sensor-based gait dataset. In 2019 Joint 8th International Conference on Informatics, Electronics & Vision (ICIEV) and 2019 3rd International Conference on Imaging, Vision & Pattern Recognition (icIVPR) (pp. 371-376). IEEE.
            - ['Pathan']:
                References: 
                [1] Pathan, R. K., Uddin, M. A., Nahar, N., Ara, F., Hossain, M. S., & Andersson, K. (2020, December). Gender classification from inertial sensor-based gait dataset. In International Conference on Intelligent Computing & Optimization (pp. 583-596). Cham: Springer International Publishing.
            - ['Sun']:
                References: 
                [1] Sun, Y., Lo, F. P. W., & Lo, B. (2019, May). A deep learning approach on gender and age recognition using a single inertial sensor. In 2019 IEEE 16th international conference on wearable and implantable body sensor networks (BSN) (pp. 1-4). IEEE.
            - ['Ahad']:
                References: 
                [1] Ahad, M. A. R., Ngo, T. T., Antar, A. D., Ahmed, M., Hossain, T., Muramatsu, D., ... & Yagi, Y. (2020). Wearable sensor-based gait analysis for age and gender estimation. Sensors, 20(8), 2424.
            - ['Mostafa']:
                References: 
                [1] Mostafa, A., Elsagheer, S. A., & Gomaa, W. (2021, July). BioDeep: A Deep Learning System for IMU-based Human Biometrics Recognition. In ICINCO (pp. 620-629).
            - ['Davarci']:
                References: 
                [1] Davarci, E., & Anarim, E. (2023). Gender Detection Based on Gait Data: A Deep Learning Approach With Synthetic Data Generation and Continuous Wavelet Transform. IEEE Access.
            

    Params.task: Task of age regression, age classification, or gender classification. 
                 ['age_regression', 'age_classification', 'gender_classification'].

    Returns
    -------

    """
    random.seed(Params.seed)

    if (Params.method == 'Khabir') or (Params.method == 'Pathan'):  # Khabir et al.'s method.
        # Set ML models.
        if Params.method == 'Pathan':
            list_name_model = ['Bagging']
        else:
            if (Params.task == 'age_classification') or (Params.task == 'gender_classification'):
                list_name_model = ['SVM']
            else:
                list_name_model = ['DT']
        # Loop over all ML models.
        for name_model in list_name_model:
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
                    if Params.method == 'Khabir':
                        model = eng.fitrtree(X_train, Y_train_target, 'MinLeafSize', 4)
                    else:
                        model = eng.fitrensemble(X_train, Y_train_target, 'Method', 'Bag', 'Learners', 'tree')
                    
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
                    if Params.method == 'Khabir':
                        t = eng.templateSVM('Standardize', True, 'KernelFunction', 'polynomial', 'type', 'Classification')
                        model = eng.fitcecoc(X_train, Y_train_target, 'Learners', t, 'FitPosterior', True)
                    else:
                        model = eng.fitcensemble(X_train, Y_train_target, 'Method', 'Bag', 'Learners', 'tree')
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
        
    elif (Params.method == 'Sun'):  # Sun et al.'s method.
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
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'MAE_train', 'MAE_test', 
                                               'RMSE_train', 'RMSE_test', 'PCC_train', 'PCC_test'])
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
                model = util_benchmark.CNN_Sun(num_channel=6, num_target=1)
                age_train = df_IMU_train.loc[:, ['age']].values.squeeze()  # Training set.
                age_test = df_IMU_test.loc[:, ['age']].values.squeeze()  # Test set.
                age_test = torch.tensor(age_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.MSELoss()
            elif Params.task == 'age_classification':  # Age group classification.
                # Load model.
                model = util_benchmark.CNN_Sun(num_channel=6, num_target=len(Params.range_age)+1)
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.squeeze()  # Training set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.squeeze()  # Test set.
                age_label_test = torch.tensor(age_label_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            elif Params.task == 'gender_classification':
                # Load model.
                model = util_benchmark.CNN_Sun(num_channel=6, num_target=2)
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
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
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

    elif Params.method == 'Ahad':  # Ahad et al.'s method.
        # Collect AE-GDI plots.
        list_condition = [0, 1]
        df_AEGDI = pd.DataFrame([])
        for condition in list_condition:
            dir_AEGDI_tmp = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'AE_GDI', 'AEGDI_con_'+str(condition)+'.h5')
            df_AEGDI_tmp = pd.read_hdf(dir_AEGDI_tmp, key='df', mode='r')
            df_AEGDI = pd.concat((df_AEGDI, df_AEGDI_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_AEGDI.index.values.tolist(), 
                                                 list_age=df_AEGDI['age'].values.tolist(), 
                                                 list_gender=df_AEGDI['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_AEGDI = df_AEGDI.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'MAE_train', 'MAE_test', 
                                               'RMSE_train', 'RMSE_test', 'PCC_train', 'PCC_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_AEGDI['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_AEGDI.loc[(df_AEGDI['age'].values >= Params.range_age[i_age]) & 
                               (df_AEGDI['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_AEGDI = df_AEGDI.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_AEGDI = df_AEGDI.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_AEGDI_train = df_AEGDI.drop(list_idx_split[fold_tmp], axis=0)
            df_AEGDI_train = pd.DataFrame(columns=df_AEGDI.columns)
            for i_fold in range(0, len(list_idx_split)):
                if i_fold == fold_tmp:
                    continue
                else:
                    df_AEGDI_train = pd.concat([df_AEGDI_train, df_AEGDI.loc[list_idx_split[i_fold], :]])
            # Gyroscope.
            data_train_gyro = df_AEGDI_train.loc[:, ['gyro_'+str(i_f) for i_f in range(0, Params.len_segment*Params.delay_max_AEGDI)]].values
            data_train_gyro = np.reshape(data_train_gyro, [data_train_gyro.shape[0], 1, Params.len_segment, Params.delay_max_AEGDI])
            # Accelerometer.
            data_train_acc = df_AEGDI_train.loc[:, ['acc_'+str(i_f) for i_f in range(0, Params.len_segment*Params.delay_max_AEGDI)]].values
            data_train_acc = np.reshape(data_train_acc, [data_train_acc.shape[0], 1, Params.len_segment, Params.delay_max_AEGDI])
            data_train = np.concatenate((data_train_gyro, data_train_acc), axis=1)
            # Test set.
            df_AEGDI_test = df_AEGDI.loc[list_idx_split[fold_tmp], :]
            # Gyroscope.
            data_test_gyro = df_AEGDI_test.loc[:, ['gyro_'+str(i_f) for i_f in range(0, Params.len_segment*Params.delay_max_AEGDI)]].values
            data_test_gyro = np.reshape(data_test_gyro, [data_test_gyro.shape[0], 1, Params.len_segment, Params.delay_max_AEGDI])
            # Accelerometer.
            data_test_acc = df_AEGDI_test.loc[:, ['acc_'+str(i_f) for i_f in range(0, Params.len_segment*Params.delay_max_AEGDI)]].values
            data_test_acc = np.reshape(data_test_acc, [data_test_acc.shape[0], 1, Params.len_segment, Params.delay_max_AEGDI])
            data_test = np.concatenate((data_test_gyro, data_test_acc), axis=1)
            # Move tensors to the target device.
            data_test = torch.tensor(data_test).to(Params.device)
            
            if Params.task == 'age_regression':  # Age regression.
                # Load model.
                model = util_benchmark.CNN_Van(num_channel=2, num_target=1)
                age_train = df_AEGDI_train.loc[:, ['age']].values.squeeze()  # Training set.
                age_test = df_AEGDI_test.loc[:, ['age']].values.squeeze()  # Test set.
                age_test = torch.tensor(age_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.MSELoss(reduction='mean')
            elif Params.task == 'age_classification':  # Age group classification.
                # Load model.
                model = util_benchmark.CNN_Van(num_channel=2, num_target=len(Params.range_age)+1)
                age_label_train = df_AEGDI_train.loc[:, ['age_label']].values.squeeze()  # Training set.
                age_label_test = df_AEGDI_test.loc[:, ['age_label']].values.squeeze()  # Test set.
                age_label_test = torch.tensor(age_label_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            elif Params.task == 'gender_classification':
                # Load model.
                model = util_benchmark.CNN_Van(num_channel=2, num_target=2)
                gender_train = df_AEGDI_train.loc[:, ['gender']].values.squeeze()  # Training set.
                gender_test = df_AEGDI_test.loc[:, ['gender']].values.squeeze()  # Test set.
                gender_test = torch.tensor(gender_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            else:
                # Other conditions. Stop training.
                sys.exit()
            model.train()
            model = model.to(Params.device)
            # Optimizer.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            # Start training.
            epochs = 1000
            for epoch in tqdm(range(epochs)):
                # Zero gradient.
                optimizer.zero_grad()
                # Select a subset from the training data.
                list_idx = np.random.choice(range(0, len(df_AEGDI_train)), 300)
                data_train_tmp = data_train[list_idx]  # Training data.
                data_train_tmp = torch.tensor(data_train_tmp).to(Params.device)  # Move the training data to target device.

                if Params.task == 'age_regression':
                    # Target age (ground truth).
                    age_train_tmp = age_train[list_idx]
                    age_train_tmp = torch.tensor(age_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_train_tmp)
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
                    age_pred_tmp = model(data_train_tmp)
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
                    age_pred_tmp = model(data_test)
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
                    
                    gender_train_tmp = gender_train[list_idx]
                    gender_train_tmp = torch.tensor(gender_train_tmp).to(Params.device)
                    # Predicted gender.
                    gender_pred_tmp = model(data_train_tmp)
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
                    gender_pred_tmp = model(data_test)
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

    elif Params.method == 'Mostafa':  # Mostafa et al.'s method.
        # Collect all segmented data.
        list_condition = [0, 1]
        df_feature_autocorr = pd.DataFrame([])
        for condition in list_condition:
            dir_feature_autocorr = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'feature_autocorr', 
                                   'feature_autocorr_con_'+str(condition)+'.csv')
            df_feature_autocorr_tmp = pd.read_csv(dir_feature_autocorr)
            df_feature_autocorr = pd.concat((df_feature_autocorr, df_feature_autocorr_tmp), ignore_index=True)
        # N-fold cross-validation.
        list_idx_split = util_data.split_balance(list_idx=df_feature_autocorr.index.values.tolist(), 
                                                 list_age=df_feature_autocorr['age'].values.tolist(), 
                                                 list_gender=df_feature_autocorr['gender'].values.tolist(), 
                                                 range_age=Params.range_age, 
                                                 num_fold=Params.num_fold, 
                                                 seed=Params.seed)
        # History of training and testing.
        if Params.task == 'age_regression':
            df_feature_autocorr = df_feature_autocorr.drop(['ID', 'condition', 'num_seq', 'gender'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'MAE_train', 'MAE_test'])
        elif Params.task == 'age_classification':
            # Create age label.
            df_feature_autocorr['age_label'] = 0
            for i_age in range(0, len(Params.range_age)-1):
                df_feature_autocorr.loc[(df_feature_autocorr['age'].values >= Params.range_age[i_age]) & 
                                        (df_feature_autocorr['age'].values < Params.range_age[i_age+1]), 'age_label'] = i_age
            df_feature_autocorr = df_feature_autocorr.drop(['ID', 'condition', 'num_seq', 'gender', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        elif Params.task == 'gender_classification':
            df_feature_autocorr = df_feature_autocorr.drop(['ID', 'condition', 'num_seq', 'age'], axis=1)
            df_history = pd.DataFrame(columns=['fold', 'epoch', 'Loss_train', 'Loss_test', 'Acc_train', 'Acc_test'])
        else:
            sys.exit()
        # Loop over each fold.
        for fold_tmp in tqdm(range(0, Params.num_fold)):
            # Train-test split.
            # Training set.
            df_feature_autocorr_train = pd.DataFrame(columns=df_feature_autocorr.columns)
            for i_fold in range(0, len(list_idx_split)):
                    if i_fold == fold_tmp:
                        continue
                    else:
                        df_feature_autocorr_train = pd.concat([df_feature_autocorr_train, df_feature_autocorr.loc[list_idx_split[i_fold], :]])
            # Input autocorrelation feature.
            data_train_autocorr = df_feature_autocorr_train.loc[:, ['f_'+str(i_f) for i_f in range(0, 6*Params.delay_max_AutoCorrFeature)]].values
            data_train_autocorr = data_train_autocorr[:, np.newaxis, :]
            # Test set.
            df_feature_autocorr_test = df_feature_autocorr.loc[list_idx_split[fold_tmp], :]
            # Input autocorrelation feature.
            data_test_autocorr = df_feature_autocorr_test.loc[:, ['f_'+str(i_f) for i_f in range(0, 6*Params.delay_max_AutoCorrFeature)]].values
            data_test_autocorr = data_test_autocorr[:, np.newaxis, :]
            # Move tensors to the target device.
            data_test_autocorr = torch.tensor(data_test_autocorr).to(Params.device)
            
            if Params.task == 'age_regression':  # Age regression.
                # Load model.
                model = util_benchmark.CNN_Mostafa(num_channel=1, num_target=1)
                age_train = df_feature_autocorr_train.loc[:, ['age']].values.squeeze()  # Training set.
                age_test = df_feature_autocorr_test.loc[:, ['age']].values.squeeze()  # Test set.
                age_test = torch.tensor(age_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.MSELoss()
            elif Params.task == 'age_classification':  # Age group classification.
                # Load model.
                model = util_benchmark.CNN_Mostafa(num_channel=1, num_target=len(Params.range_age)+1)
                age_label_train = df_feature_autocorr_train.loc[:, ['age_label']].values.squeeze()  # Training set.
                age_label_test = df_feature_autocorr_test.loc[:, ['age_label']].values.squeeze()  # Test set.
                age_label_test = torch.tensor(age_label_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            elif Params.task == 'gender_classification':
                # Load model.
                model = util_benchmark.CNN_Mostafa(num_channel=1, num_target=2)
                gender_train = df_feature_autocorr_train.loc[:, ['gender']].values.squeeze()  # Training set.
                gender_test = df_feature_autocorr_test.loc[:, ['gender']].values.squeeze()  # Test set.
                gender_test = torch.tensor(gender_test, dtype=torch.float32).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            else:
                # Other conditions. Stop training.
                sys.exit()
            model.train()
            model = model.to(Params.device)
            # Optimizer.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # Start training.
            epochs = 5000
            for epoch in tqdm(range(epochs)):
                # Zero gradient.
                optimizer.zero_grad()
                # Select a subset from the training data.
                list_idx = np.random.choice(range(0, len(df_feature_autocorr_train)), 300)
                data_train_autocorr_tmp = data_train_autocorr[list_idx]  # Training data.
                data_train_autocorr_tmp = torch.tensor(data_train_autocorr_tmp).to(Params.device)  # Move the training data to target device.

                if Params.task == 'age_regression':
                    # Target age (ground truth).
                    age_train_tmp = age_train[list_idx].astype(dtype=np.float32)
                    age_train_tmp = torch.tensor(age_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_train_autocorr_tmp.type(torch.float32))
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
                    age_pred_tmp = model(data_test_autocorr.type(torch.float32))
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
                    age_pred_tmp = model(data_train_autocorr_tmp.type(torch.float32))
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
                    age_pred_tmp = model(data_test_autocorr.type(torch.float32))
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
                    gender_pred_tmp = model(data_train_autocorr_tmp.type(torch.float32))
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
                    gender_pred_tmp = model(data_test_autocorr.type(torch.float32))
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
            
    elif Params.method == 'Davarci':  # Davarci et al.'s method.
        # Collect all CWT feature data.
        list_condition = [0, 1]
        df_IMU = pd.DataFrame([])
        for condition in list_condition:
            # Information dataframe.
            dir_IMU = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 
                                   'data_segmented_con_'+str(condition)+'.csv')
            df_IMU_tmp = pd.read_csv(dir_IMU)
            df_IMU_tmp = df_IMU_tmp.loc[:, ['ID', 'condition', 'num_seq', 'age', 'gender']]
            df_IMU = pd.concat((df_IMU, df_IMU_tmp), ignore_index=True)
            # CWT feature.
            dir_cwt = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'CWT', 
                                   'CWT_con_'+str(condition)+'.npy')
            data_cwt_tmp = np.load(dir_cwt)
            if condition == 0:
                data_cwt = data_cwt_tmp
            else:
                data_cwt = np.concatenate((data_cwt, data_cwt_tmp), axis=0)
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
            data_cwt_train = np.empty((0, 8, 64, 128))
            df_IMU_train = pd.DataFrame(columns=df_IMU.columns)
            for i_fold in range(0, len(list_idx_split)):
                if i_fold == fold_tmp:
                    data_cwt_test = data_cwt[list_idx_split[i_fold]]
                    df_IMU_test = df_IMU.loc[list_idx_split[i_fold], :].copy()
                else:
                    data_cwt_train = np.concatenate((data_cwt_train, data_cwt[list_idx_split[i_fold]]), axis=0)
                    df_IMU_train = pd.concat([df_IMU_train, df_IMU.loc[list_idx_split[i_fold], :].copy()])
            # Move test data to the target device.
            data_cwt_test = torch.tensor(data_cwt_test).to(Params.device)
            
            if Params.task == 'age_regression':  # Age regression.
                # Load model.
                model = util_benchmark.CNN_Davarci(num_channel=8, num_target=1)
                age_train = df_IMU_train.loc[:, ['age']].values.squeeze()  # Training set.
                age_test = df_IMU_test.loc[:, ['age']].values.squeeze()  # Test set.
                age_test = torch.tensor(age_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.MSELoss()
            elif Params.task == 'age_classification':  # Age group classification.
                # Load model.
                model = util_benchmark.CNN_Davarci(num_channel=8, num_target=len(Params.range_age)+1)
                age_label_train = df_IMU_train.loc[:, ['age_label']].values.squeeze()  # Training set.
                age_label_test = df_IMU_test.loc[:, ['age_label']].values.squeeze()  # Test set.
                age_label_test = torch.tensor(age_label_test).to(Params.device)
                # Criterion.
                criterion = torch.nn.CrossEntropyLoss()
            elif Params.task == 'gender_classification':
                # Load model.
                model = util_benchmark.CNN_Davarci(num_channel=8, num_target=2)
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
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            # Start training.
            epochs = 1000
            for epoch in tqdm(range(epochs)):
                # Zero gradient.
                optimizer.zero_grad()
                # Select a subset from the training data.
                list_idx = np.random.choice(range(0, len(df_IMU_train)), 300)
                data_cwt_train_tmp = data_cwt_train[list_idx]  # Training data.
                data_cwt_train_tmp = torch.tensor(data_cwt_train_tmp).to(Params.device)  # Move the training data to target device.

                if Params.task == 'age_regression':
                    # Target age (ground truth).
                    age_train_tmp = age_train[list_idx].astype(dtype=np.float32)
                    age_train_tmp = torch.tensor(age_train_tmp).to(Params.device)
                    # Predicted age.
                    age_pred_tmp = model(data_cwt_train_tmp.type(torch.float32))
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
                    age_pred_tmp = model(data_cwt_test.type(torch.float32))
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
                    age_pred_tmp = model(data_cwt_train_tmp.type(torch.float32))
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
                    age_pred_tmp = model(data_cwt_test.type(torch.float32))
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
                    gender_pred_tmp = model(data_cwt_train_tmp.type(torch.float32))
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
                    gender_pred_tmp = model(data_cwt_test.type(torch.float32))
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


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')   # Load pre-defiend options.
    name_dataset = 'OU_ISIR_Inertial_Sensor'   # ['OU_ISIR_Inertial_Sensor'].
    Params = util_data.Params(dir_option, name_dataset)   # Initialize parameter object.
    list_method = ['Khabir', 'Pathan', 'Sun', 'Ahad', 'Mostafa', 'Davarci']
    list_type_imu = ['manual_IMUZLeft', 'manual_IMUZCenter', 'manual_IMUZRight']
    list_task = ['age_classification', 'gender_classification', 'age_regression']
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        for task in list_task:
            Params.task = task
            for method in list_method:
                Params.method = method
                # Print current model information.
                print('Method: '+method+'. Data Type: '+type_imu+'. Task: '+task+'.')
                main_Benchmark_previous(Params=Params)