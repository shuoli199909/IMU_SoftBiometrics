# Author: Shuo Li
# Date: 2024/01/09

import os
import sys
import tqdm
import numpy as np
import pandas as pd
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data


def main_IMU2ManualFeature(Params):
    """Main function of transforming segmented IMU data into manually designed features. Save the features into csv files.

    References
    ----------
    [1] Khabir, K. M., Siraj, M. S., Ahmed, M., & Ahmed, M. U. (2019, May). Prediction of gender and age from inertial sensor-based gait dataset. In 2019 Joint 8th International Conference on Informatics, Electronics & Vision (ICIEV) and 2019 3rd International Conference on Imaging, Vision & Pattern Recognition (icIVPR) (pp. 371-376). IEEE.
    [2] Pathan, R. K., Uddin, M. A., Nahar, N., Ara, F., Hossain, M. S., & Andersson, K. (2020, December). Gender classification from inertial sensor-based gait dataset. In International Conference on Intelligent Computing & Optimization (pp. 583-596). Cham: Springer International Publishing.

    Parameters
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """

    # Get current directory.
    dir_crt = os.getcwd()
    # Collect all segmented data.
    dir_df_seg = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 'data_segmented_con_'+str(Params.condition)+'.csv')
    df_seg = pd.read_csv(dir_df_seg)
    # Loop over all data segments.
    for i_seg in tqdm.tqdm(range(0, len(df_seg))):
        # Load segmented data.
        df_data = df_seg.loc[i_seg, :].copy()
        # Load ID, age, and gender information.
        ID = int(df_data['ID'])
        condition = int(df_data['condition'])
        num_seq = int(df_data['num_seq'])
        age = int(df_data['age'])
        gender = int(df_data['gender'])
        # Collect IMU data.
        data_gyro_x = df_data[['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values  # Gyroscope x-axis.
        data_gyro_y = df_data[['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values  # Gyroscope y-axis.
        data_gyro_z = df_data[['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values  # Gyroscope z-axis.
        data_acc_x = df_data[['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values  # Accelerometer x-axis.
        data_acc_y = df_data[['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values  # Accelerometer y-axis.
        data_acc_z = df_data[['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values  # Accelerometer z-axis.
        data_imu = np.row_stack((data_gyro_x, data_gyro_y, data_gyro_z, data_acc_x, data_acc_y, data_acc_z))  # IMU data.
        # Extract manually designed features.
        feature_manual = util_data.IMU2ManualFeature(data_imu)
        feature_manual = pd.concat((feature_manual, 
                                    pd.Series({'ID': ID, 'condition': condition, 'num_seq': num_seq, 'age': age, 'gender': gender})))
        if i_seg == 0:
            # Create new dataframe to save data.
            df_feature_manual = feature_manual.to_frame().T
        else:
            df_feature_manual = pd.concat((df_feature_manual, feature_manual.to_frame().T), ignore_index=True)
    # Save extracted features.
    dir_save = os.path.join(dir_crt, 'data', name_dataset, Params.type_imu, 'feature_manual', 'feature_manual_con_'+str(Params.condition)+'.csv')
    df_feature_manual.to_csv(dir_save, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')  # Load pre-defiend options.
    name_dataset = 'OU_ISIR_Inertial_Sensor'  # ['OU_ISIR_Inertial_Sensor', 'OU_ISIR_Similar_Action'].
    Params = util_data.Params(dir_option, name_dataset)  # Initialize parameter object.
    list_type_imu = ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    list_condition = [0, 1]
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        for condition in list_condition:
            Params.condition = condition
            # Print current data information.
            print('Data Type: '+type_imu+'. Walk Condition: '+str(condition)+'.')
            main_IMU2ManualFeature(Params=Params)