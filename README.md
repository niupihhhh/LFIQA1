

**Multi-task Guided Blind Light Field Image Quality Assessment via Spatial-Frequency Collaborative Modeling**
**Note: First, we convert the dataset into H5 files using MATLAB. Then, we train and test the model in Python.**

### Generate Dataset in MATLAB
Take the NBU-LF1.0 dataset for instance, convert the dataset into h5 files, and then put them into './Datasets/NBU_MLI_7x32x32/':
```
 ./MAFBLiF/Datasets/Generateh5_for_NBU_Dataset.m
```

### Train
Train the model using the following command:
```
python Train.py  --trainset_dir ./Datasets/NBU_MLI_7x32x32/
```

### Test Overall Performance
Test the overall performance using the following command:
```
python Test.py
```

### Test Individual Distortion Type Performance
Test the individual distortion type performance using the following command:
```
 python Test_Dist.py
```
### Acknowledgement
This project is based on [DeeBLiF](https://github.com/ZhengyuZhang96/DeeBLiF). Thanks for the awesome work.

### Citation
Please cite the following paper if you use this repository in your reseach.
```

