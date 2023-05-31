#Uncomment the following if you don't have the kitti data. Check the project repo for the correct repo structure
#wget https://huggingface.co/LXT/VideoK-Net/resolve/main/kitti_step.zip
#apt-get update
#apt-get install unzip
#unzip kitti_step.zip

#Get the weights with the other scripts


# 1. Create a conda virtual environment.
conda create -n prova_da_sh python=3.7 -y
source activate prova_da_sh
echo $CONDA_PREFIX

# 2. Install PyTorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV and MMDetection
pip install openmim
mim install mmcv-full==1.3.14
mim install mmdet==2.18.0

# 4. Install other dependencies
pip install git+https://github.com/cocodataset/panopticapi.git
pip install timm==0.6.13
pip install imageio==2.6.1
pip install scipy==1.7.3
pip install lap==0.4.0
pip install Cython==0.29.34
pip install cython-bbox==0.1.3
pip install seaborn==0.12.2


#command to start training of pre-trained model. The model is pre-trained on cityscapes, the training is on kitti. 

#bash ./tools/dist_train.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $NUM_OF_GPUS $FOLDER_FOR_RESULTS --no-validate --load-from $FOLDER_FOR_WEIGHTS







