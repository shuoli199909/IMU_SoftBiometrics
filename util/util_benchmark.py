"""
Utils for the experiments of existing algorithms (benchmark).
"""

# Author: Shuo Li
# Date: 2024/01/29

import torch


class CNN_Sun(torch.nn.Module):
    """1D-CNN model for age and gender estimation proposed by Sun et al.
    """

    def __init__(self, num_channel=6, num_target=2) -> None:
        """Groundtruth class initialization.

        References
        ----------
        [1] Sun, Y., Lo, F. P. W., & Lo, B. (2019, May). A deep learning approach on gender and age recognition using a single inertial sensor. In 2019 IEEE 16th international conference on wearable and implantable body sensor networks (BSN) (pp. 1-4). IEEE.

        Parameters
        ----------
        num_channel: Number of channels of the input data. size=[1].
        num_target: Number of output targets. size=[1].

        Returns
        -------

        """

        super(CNN_Sun, self).__init__()
        filterNum_1 = 200
        filterNum_2 = 400
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=num_channel, out_channels=filterNum_1, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=3)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filterNum_1, out_channels=filterNum_1, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filterNum_1, out_channels=filterNum_2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filterNum_2, out_channels=filterNum_2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.dropout = torch.nn.Dropout(0.05)
        self.fc1 = torch.nn.Linear(1200, 3000)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(3000, num_target)
    
    def forward(self, x):
        x = x.to(torch.float32)
        # 1D CNN.
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        # MLP.
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class CNN_Mostafa(torch.nn.Module):
    """1D-CNN model for age and gender estimation proposed by Mostafa et al.
    """

    def __init__(self, num_channel=1, num_target=2) -> None:
        """Groundtruth class initialization.

        References
        ----------
        [1] Mostafa, A., Elsagheer, S. A., & Gomaa, W. (2021, July). BioDeep: A Deep Learning System for IMU-based Human Biometrics Recognition. In ICINCO (pp. 620-629).

        Parameters
        ----------
        num_channel: Number of channels of the input data. size=[1].
        num_target: Number of output targets. size=[1].

        Returns
        -------

        """

        super(CNN_Mostafa, self).__init__()
        filterNum_1 = 16
        filterNum_2 = 32
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=num_channel, out_channels=filterNum_1, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filterNum_1, out_channels=filterNum_2, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = torch.nn.Linear(256, 64)
        self.dropout = torch.nn.Dropout(0.5)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, num_target)
    
    def forward(self, x):
        x = x.to(torch.float32)
        # 1D CNN.
        x = self.layer_1(x)
        x = self.layer_2(x)
        # MLP.
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class CNN_Van(torch.nn.Module):
    """2D-CNN model for age and gender estimation.
    """

    def __init__(self, num_channel=2, num_target=2) -> None:
        """Groundtruth class initialization.

        References
        ----------
        [1] Van Hamme, T., Garofalo, G., Argones Rúa, E., Preuveneers, D., & Joosen, W. (2019). A systematic comparison of age and gender prediction on imu sensor-based gait traces. Sensors, 19(13), 2945.
        [2] Ahad, M. A. R., Ngo, T. T., Antar, A. D., Ahmed, M., Hossain, T., Muramatsu, D., ... & Yagi, Y. (2020). Wearable sensor-based gait analysis for age and gender estimation. Sensors, 20(8), 2424.
        [3] Zhao, Y., & Zhou, S. (2017). Wearable device-based gait recognition using angle embedded gait dynamic images and a convolutional neural network. Sensors, 17(3), 478.

        Parameters
        ----------
        num_channel: Number of channels of the input data. size=[1].
        num_target: Number of output targets. size=[1].

        Returns
        -------

        """

        
        super(CNN_Van, self).__init__()
        filterNum_1 = 32
        filterNum_2 = 64
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel, out_channels=filterNum_1, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=filterNum_1, out_channels=filterNum_2, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        self.layer_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=filterNum_2, out_channels=filterNum_2, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = torch.nn.Linear(2048, 1024)
        self.dropout = torch.nn.Dropout(0.3)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(1024, num_target)
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        # 2D CNN.
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_block(x)
        x = self.layer_block(x)
        # MLP.
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class CNN_Davarci(torch.nn.Module):
    """2D-CNN-LSTM model for age and gender estimation.
    """

    def __init__(self, num_channel=8, num_target=2) -> None:
        """Groundtruth class initialization.

        References
        ----------
        [1] Davarci, E., & Anarim, E. (2023). Gender Detection Based on Gait Data: A Deep Learning Approach With Synthetic Data Generation and Continuous Wavelet Transform. IEEE Access.

        Parameters
        ----------
        num_channel: Number of channels of the input data. size=[1].
        num_target: Number of output targets. size=[1].

        Returns
        -------

        """

        
        super(CNN_Davarci, self).__init__()
        filterNum_1 = 64
        filterNum_2 = 32
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel, out_channels=filterNum_1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=filterNum_1, out_channels=filterNum_1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=filterNum_1, out_channels=filterNum_2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = torch.nn.LSTM(4096, 64)
        self.dropout = torch.nn.Dropout(p=0.15)
        self.fc1 = torch.nn.Linear(64, 1000)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(1000, num_target)
    
    def forward(self, x):
        x = x.to(torch.float32)
        # 2D CNN.
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # MLP.
        x = x.contiguous().view(x.size(0), -1)
        x = self.lstm(x)[0]
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)

        return x