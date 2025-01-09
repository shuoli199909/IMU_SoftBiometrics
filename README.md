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
- matlabengine==23.2
- numpy==1.26.4
- pandas==2.2.2
- PyYAML==6.0.1
- scikit_learn==1.4.1.post1
- scipy==1.13.1
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
| Combined Feature \+ KNN             | 91\.93\%                     |
| Combined Feature \+ NB              | 48\.50\%                     |
| Combined Feature \+ SVM             | 80\.24\%                     |
| Combined Feature \+ DT              | 49\.66\%                     |
| Combined Feature \+ BT              | 77\.31\%                     |
| Combined Feature \+ DA              | 57\.39\%                     |
| InceptionTime                       | 57\.39\%                     |
| H-InceptionTime                     | 95\.44\%                     |
| ROCKET                              | 78\.93\%                     |
| MiniROCKET                          | 77\.25\%                     |
| MultiROCKET                         | 68\.66\%                     |
| HYDRA                               | 91\.02\%                     |
### Gender Estimation (Average)
| **Method**                          | **Gender Classification**                 |
|:-----------------------------------:|:-----------------------------:|
| Combined Feature \+ KNN             | 95\.69\%                     |
| Combined Feature \+ NB              | 71\.04\%                     |
| Combined Feature \+ SVM             | 81\.09\%                     |
| Combined Feature \+ DT              | 73\.07\%                     |
| Combined Feature \+ BT              | 86\.10\%                     |
| Combined Feature \+ DA              | 79\.69\%                     |
| InceptionTime                       | 98\.62\%                     |
| H-InceptionTime                     | 98\.61\%                     |
| ROCKET                              | 89\.86\%                     |
| MiniROCKET                          | 89\.13\%                     |
| MultiROCKET                         | 84\.50\%                     |
| HYDRA                               | 96\.07\%                     |
### Age Value Regression (Average)
| **Method**                          | **Age Value Regression (MAE)**        |
|:-----------------------------------:|:-------------------------:|
| Combined Feature \+ GPR             | 5.63                     |
| Combined Feature \+ LR              | 17.08                     |
| Combined Feature \+ SVM             | 7.22                     |
| Combined Feature \+ DT              | 11.37                     |
| Combined Feature \+ BT              | 9.23                     |
| InceptionTime                       | 3.54                     |
| H-InceptionTime                     | 3.40                     |
| ROCKET                              | 8.60                     |
| MiniROCKET                          | 8.08                     |
| MultiROCKET                         | 9.16                     |
| HYDRA                               | 6.11                     |
### Contact
If you have any questions, please feel free to contact me through email (shuoli199909@outlook.com)!
## License - MIT License.
