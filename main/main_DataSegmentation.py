"""
Main function of segmenting raw data sequence using sliding windows. Create segmented data for further training and testing.
"""

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


def main_DataSegmentation(Params):
    """
    Parameters 
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """

    # load data.
    GT = util_data.GroundTruth(Params=Params)
    # Load the ID info csv.
    info_raw = pd.read_csv(Params.dir_info)
    # Initialize dataframe.
    df_seg = pd.DataFrame(columns=['ID', 'condition', 'num_seq', 'age', 'gender'])  # Subject information.
    df_seg = pd.concat((df_seg, 
                        pd.DataFrame(columns=['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Gyroscope x-axis.
                        pd.DataFrame(columns=['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Gyroscope y-axis.
                        pd.DataFrame(columns=['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Accelerometer z-axis.
                        pd.DataFrame(columns=['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Accelerometer x-axis.
                        pd.DataFrame(columns=['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Accelerometer y-axis.
                        pd.DataFrame(columns=['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0])  # Accelerometer z-axis.
                        ), ignore_index=True)
    df_seg.drop(df_seg.index, inplace=True)
    # Collect all segmented data.
    # Loop over all subjects.
    for ID in tqdm.tqdm(info_raw['ID']):
        # Collect IMU data and groundtruth.
        gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, age, gender = GT.get_GT(specification=[Params.type_imu, Params.condition, ID])
        num_seq = 0
        if np.isnan(gyro_x).all():  # No accessible data.
            continue
        else:
            while len(gyro_x) >= Params.len_segment:
                # Current data collection.
                df_tmp = pd.DataFrame(columns=df_seg.columns.values.tolist())
                # Segementation using a sliding window of [Params.len_segment] samples.
                df_tmp.loc[0, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]] = gyro_x[:Params.len_segment]
                df_tmp.loc[0, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]] = gyro_y[:Params.len_segment]
                df_tmp.loc[0, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]] = gyro_z[:Params.len_segment]
                df_tmp.loc[0, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]] = acc_x[:Params.len_segment]
                df_tmp.loc[0, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]] = acc_y[:Params.len_segment]
                df_tmp.loc[0, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]] = acc_z[:Params.len_segment]
                # Move the sliding window.
                gyro_x = gyro_x[Params.len_slide:]
                gyro_y = gyro_y[Params.len_slide:]
                gyro_z = gyro_z[Params.len_slide:]
                acc_x = acc_x[Params.len_slide:]
                acc_y = acc_y[Params.len_slide:]
                acc_z = acc_z[Params.len_slide:]
                # Save processed data.
                df_tmp.loc[0, ['ID', 'condition', 'num_seq', 'age', 'gender']] = [ID, Params.condition, num_seq, age, gender]
                df_seg = pd.concat((df_seg, df_tmp), ignore_index=True)
                # Next window.
                num_seq = num_seq + 1
    # Save dataframe.
    dir_save = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 'data_segmented', 'data_segmented_con_'+str(Params.condition)+'.csv')
    df_seg.to_csv(dir_save, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZLeft', 'manual_IMUZCenter', 'manual_IMUZRight']
    list_condition = [0, 1]
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        for condition in list_condition:
            Params.condition = condition
            # Print current data information.
            print('Data Type: '+type_imu+'. Walk Condition: '+str(condition)+'.')
            main_DataSegmentation(Params=Params)