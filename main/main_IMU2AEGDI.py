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


def main_IMU2AEGDI(Params):
    """Main function of transforming segmented IMU data into AE-GDI plots. Save the 2D plots using Numpy arrays.

    References
    ----------
    [1] Zhao, Y., & Zhou, S. (2017). Wearable device-based gait recognition using angle embedded gait dynamic images and a convolutional neural network. Sensors, 17(3), 478.
    [2] Van Hamme, T., Garofalo, G., Argones Rúa, E., Preuveneers, D., & Joosen, W. (2019). A systematic comparison of age and gender prediction on imu sensor-based gait traces. Sensors, 19(13), 2945.

    Parameters
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """

    # Get current directory.
    dir_crt = os.getcwd()
    # Ignore invalid division.
    np.seterr(divide='ignore',invalid='ignore')
    # Create feature dataframe.
    df_AEGDI = pd.DataFrame([])
    # Create IMU data segment dataframe (original).
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
        AEGDI = util_data.IMU2AEGDI(data_imu, Params.delay_max_AEGDI)
        AEGDI = pd.concat((pd.Series({'ID': ID, 'condition': condition, 'num_seq': num_seq, 'age': age, 'gender': gender}), 
                           AEGDI))
        df_AEGDI = pd.concat((df_AEGDI, AEGDI.to_frame().T), ignore_index=True)
        # Save extracted features.
        dir_save = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'AE_GDI', 'AEGDI_con_'+str(Params.condition)+'.h5')
        df_AEGDI.to_hdf(dir_save, key='df')


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')  # Load pre-defiend options.
    name_dataset = 'OU_ISIR_Inertial_Sensor'  # ['OU_ISIR_Inertial_Sensor', 'OU_ISIR_Similar_Action'].
    Params = util_data.Params(dir_option, name_dataset)  # Initialize parameter object.
    list_type_imu = ['auto_IMUZCenter', 'manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    list_condition = [0, 1]
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        for condition in list_condition:
            Params.condition = condition
            # Print current data information.
            print('Data Type: '+type_imu+'. Walk Condition: '+str(condition)+'.')
            main_IMU2AEGDI(Params=Params)