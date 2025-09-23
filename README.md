Multi-task Guided Blind Light Field Image Quality Assessment via Spatial-Frequency Collaborative Modeling

**Note:**
 This repository provides the implementation of our NR-LFIQA method.
 We first convert the dataset into H5 files using MATLAB, and then train and evaluate the model in Python.

------

## 1. Generate Dataset in MATLAB

Take the **NBU-LF1.0** dataset as an example.
 Convert the dataset into H5 files and place them under `./Datasets/NBU_MLI_7x32x32/`:

```
./Datasets/Generateh5_for_NBU_proposed.m
```

- `Generateh5_for_NBU_proposed.m` creates multi-level light-field image (MLI) patches required for training and testing.

------

## 2. Train

Train the proposed model with:

```
python Train.py --trainset_dir ./Datasets/NBU_MLI_7x32x32/
```

Key features during training:

- **Spatial–frequency collaborative modeling** with **tensor decomposition** (TD-SAI) and **3D-DCT** (DCT-PVS) representations.
- **Multi-task learning**: simultaneous prediction of quality score and distortion type.

------

## 3. Test Overall Performance

Evaluate the trained model on the test set:

```
python Test.py
```

------

## 4. Test Individual Distortion Types

Assess performance for each distortion type:

```
python Test_Dist.py
```

------

## 5. Method Overview

- **Stage 1 – Spatial-Frequency Representation**
  - Construct complementary spatial and frequency representations using **tensor decomposition** (TD-SAI) and **3D-DCT** (DCT-PVS).
- **Stage 2 – Hybrid Context-aware Convolution Module (HCCM)**
  - Combines **Hybrid Convolution (HC-Conv)** with multiple dilation rates (1, 2, 3) and **Channel Attention (CA)** to jointly capture local details and global context.
- **Stage 3 – Multi-task Prediction**
  - Outputs both the final quality score and distortion category.

------

## 6. Acknowledgement

This project is inspired by [DeeBLiF](https://github.com/ZhengyuZhang96/DeeBLiF).
 Thanks to the authors for their excellent work.
