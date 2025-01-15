# IMU_SoftBiometrics

## Introduction
This project provides the code for the experiments of IMU soft biometrics.
## Summary
This project mainly comprises five parts:
1. Segmenting the original IMU gait signals into IMU sequences of equal length. (csv format)
2. Transforming the IMU sequences into hand-crafted features. (csv format)
3. Transforming the IMU sequences into autocorrelation features. (csv format)
4. Transforming the IMU sequences into AE-GDI features. (h5 format)
5. Transforming the IMU sequences into CWT features. (h5 format)
6. Systematically evaluating the performance of all developed algorithms in TSC problems for soft biometrics.
7. Also evaluating the previous algorithms of IMU soft biometrics.
## Package Requirements
- matlabengine==24.2
- numpy==1.26.0
- pandas==2.2.2
- PyYAML==6.0.1
- scikit_learn==1.4.1.post1
- scipy==1.14.0
- statsmodels==0.14.1
- torch==2.2.1+cu118
- tqdm==4.66.2
- tsai==0.3.9
## Configuration
The experiments were runned on the author's personal laptop. The configurations are provided as the reference:
- CPU: AMD Ryzen 9 5900HX with Radeon Graphics
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU
- CUDA Version: 11.7
- Operating System: Microsoft Windows 11 (version-10.0.22631)
## Code Structure
```bash
IMU_SoftBiometrics
├─config
├─data
│  └─OU_ISIR_Inertial_Sensor
│     ├─manual_IMUZCenter
│     │  ├─AE_GDI
│     │  ├─CWT
│     │  ├─data_segmented
│     │  ├─feature_autocorr
│     │  └─feature_manual
│     ├─manual_IMUZLeft
│     │  ├─AE_GDI
│     │  ├─CWT
│     │  ├─data_segmented
│     │  ├─feature_autocorr
│     │  └─feature_manual
│     └─manual_IMUZRight
│        ├─AE_GDI
│        ├─CWT
│        ├─data_segmented
│        ├─feature_autocorr
│        └─feature_manual
├─main
│  ├─main_Benchmark_previous
│  ├─main_Benchmark_developed
│  ├─main_IMU2AutoCorrFeature
│  ├─main_IMU2AEGDI
│  ├─main_IMU2CWTFeature
│  ├─main_IMU2ManualFeature
│  └─main_DataSegmentation
├─result
│  └─OU_ISIR_Inertial_Sensor
│     ├─manual_IMUZCenter
│     ├─manual_IMUZLeft
│     └─manual_IMUZRight
└─util

```
## Datasets
1. OU-ISIR Inertial Sensor Dataset: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html
2. OU-ISIR Similar Action Inertial Dataset: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/SimilarActionsInertialDB.html
## Usage
First, activate the local environment and then set the folder containing this README file as the current folder.  
For Windows, execute: **python (...).py**  
For Linux, execute: **python3 (...).py**  
1. Segment the original IMU gait signals into IMU sequences of equal length: **python "./main/main_DataSegmentation.py"**
2. Transform segmented IMU signals into hand-crafted features: **python "./main/main_IMU2ManualFeature.py"**
3. Transform segmented IMU signals into autocorrelation features: **python "./main/main_IMU2AutoCorrFeature.py"**
4. Transform segmented IMU signals into AE-GDI features: **python "./main/main_IMU2AEGDI.py"**
5. Transform segmented IMU signals into CWT features: **python "./main/main_IMU2CWTFeature.py"**
6. Transform segmented IMU signals into AE-GDI features: **python "./main/main_IMU2AEGDI.py"**
7. Evaluate the performance of included algorithms for soft biometrics: **python "./main/main_Benchmark.py"**
## Example Results
### Age Group Classification (Average)
| **Method**                          | **Age Group Classification**                 |
|:-----------------------------------:|:-----------------------------:|
| Combined Feature \+ KNN             | 85\.97\%                     |
| Combined Feature \+ NB              | 46\.92\%                     |
| Combined Feature \+ SVM             | 76\.40\%                     |
| Combined Feature \+ DT              | 46\.47\%                     |
| Combined Feature \+ BT              | 71\.32\%                     |
| Combined Feature \+ DA              | 55\.95\%                     |
| InceptionTime                       | 92\.37\%                     |
| H-InceptionTime                     | 91\.76\%                     |
| ROCKET                              | 74\.62\%                     |
| MiniROCKET                          | 71\.95\%                     |
| MultiROCKET                         | 63\.87\%                     |
| HYDRA                               | 82\.68\%                     |
### Gender Estimation (Average)
| **Method**                          | **Gender Classification**                 |
|:-----------------------------------:|:-----------------------------:|
| Combined Feature \+ KNN             | 90\.90\%                     |
| Combined Feature \+ NB              | 72\.52\%                     |
| Combined Feature \+ SVM             | 75\.78\%                     |
| Combined Feature \+ DT              | 71\.71\%                     |
| Combined Feature \+ BT              | 85\.16\%                     |
| Combined Feature \+ DA              | 80\.24\%                     |
| InceptionTime                       | 97\.07\%                     |
| H-InceptionTime                     | 96\.74\%                     |
| ROCKET                              | 87\.25\%                     |
| MiniROCKET                          | 86\.45\%                     |
| MultiROCKET                         | 82\.86\%                     |
| HYDRA                               | 93\.15\%                     |
### Age Value Regression (Average)
| **Method**                          | **Age Value Regression (MAE)**        |
|:-----------------------------------:|:-------------------------:|
| Combined Feature \+ GPR             | 6.23                     |
| Combined Feature \+ LR              | 16.80                     |
| Combined Feature \+ SVM             | 7.46                     |
| Combined Feature \+ DT              | 11.70                     |
| Combined Feature \+ BT              | 9.34                     |
| InceptionTime                       | 3.95                     |
| H-InceptionTime                     | 3.94                     |
| ROCKET                              | 8.86                     |
| MiniROCKET                          | 8.36                     |
| MultiROCKET                         | 9.20                     |
| HYDRA                               | 6.85                     |
### Contact
If you have any questions, please feel free to contact me through email (shuoli199909@outlook.com)!
## License - MIT License.
