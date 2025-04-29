# ImConDM: Accurate MTS Anomaly Detection Integrating Imputation with Conditional Diffusion Models

## ImConDM

This repository contains code for the paper, IMCONDM: ACCURATE MTS ANOMALY DETECTION INTEGRATING IMPUTATION WITH CONDITIONAL DIFFUSION MODELS.
(The code is being sorted out and we will continue to update it.)

##  Overview

This repository is the implementation of ImConDM: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection. We propose the ImConDM framework for anomaly detection and evaluate its performance on six open-source datasets.

## Datasets

1. PSM (PooledServer Metrics) is collected internally from multiple application server nodes at eBay.
2. SMAP (Soil Moisture Active Passive satellite) also is a public dataset from NASA. 
3. WADI (Water Distribution) is obtained from 127 sensors of the critical infrastructure system under continuous operations. 
4. SWAT (Secure Water Treatment) is obtained from 51 sensors of the critical infrastructure system under continuous operations. 
5. MSL (Mars Science Laboratory) datasets consist of telemetry information and sensor/actuator data from the Mars rover. The data is specifically collected for research purposes related to detecting anomalies in the rover’s operations.
6. SMD (Server Machine Dataset) comprises stacked traces of resource utilization data collected from a compute cluster within a large Internet company. Covering a five-week period, it includes data from 28 machines (subsets).

We apply our method on six datasets, the SWAT and WADI datasets, in which we did not upload data in this repository.Please refer to [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/) and send request to iTrust is you want to try the data.

## How to run

- Train and detect:

> python main.py --dataset SMD 
>
> Then you will train the whole model and will get the reconstructed data and detected score.

## How to run with your own data

- By default, datasets are placed under the "data" folder. If you need to change the dataset, you can modify the dataset path  in the main file.Then you should change the corresponding parameters of diffusion.py

> python main.py  --'dataset'  your dataset



## Result

We  use dataset PSM for testing demonstration, you can run main.py directly and get the corresponding result.
