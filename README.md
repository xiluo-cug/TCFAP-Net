# TCFAP-Net: Transformer-based Cross-domain Fusion and Adaptive Perception Network for Large-scale Point Cloud Semantic Segmentation

This is the official implementation of **TCFAP-Net**. 
 


	
### (1) Setup
This code has been tested with Python 3.5, Tensorflow 1.11, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04.
 
- Clone the repository 
```
git clone --depth=1 https://github.com/xiluo-cug/TCFAP-Net && cd TCFAP-Net
```
- Setup python environment
```
conda create -n tcfapnet python=3.5
source activate tcfapnet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Start 6-fold cross validation:
```
sh jobs_6_fold_cv_s3dis.sh
```
- Move all the generated results (*.ply) in `/test` folder to `/data/S3DIS/results`, calculate the final mean IoU results:
```
python utils/6_fold_cv.py
```


