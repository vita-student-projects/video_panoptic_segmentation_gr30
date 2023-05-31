# How to run the code

## Setting up the environment

Before doing anything, you need to setup the environment by running the setup_videoknet_v2.sh with the following command


```
bash setup_videoknet_v2.sh
```

## Training, running inference, and evaluting on kittistep

In order to train, run inference and evaluate on kittistep you need to download the kittistep dataset and put it in the `data` folder. To make sure that the data is in the right format, you should have a folder "kitti-step" structured in the following way.

```
kitti-step 
├── video_squence
│   ├──  val
│   │    ├──  000002_000000_leftImg8bit.png
│   │    ├──  000002_000000_panoptic.png
│   │    ├──  000002_000001_leftImg8bit.png
│   │    ├──  000002_000001_panoptic.png
│   │    ├──  ...
│   ├──  train
|   │    ├──  ...
│   ├──  test 
│        ├──  ...
```

These kitti-step folder has to be put inside the `data` folder.

To train the model on cityscapes starting from checkpoints pre-trained on citiscapres, you should should first of all download weights in the `weights` folder (you have to create it first). You can find our best weights (latest.pth) and the weights of the authors in this [Google Drive  folder](https://drive.google.com/drive/folders/13rXX12MUjAfj-HlwVWa_QKNuKl8UmsF9?usp=share_link).

To run the training step you should run the following command (substitute `TRAIN_OUTPUT_DIR` with the path to the working directory and `CHECKPOINT` with the path to the checkpoint):



```bash
# train Video K-Net on KITTI-step with 3 GPUs from pretrained checkpoint
bash ./tools/dist_train.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py 1 $TRAIN_OUTPUT_DIR --no-validate --load-from $CHECKPOINT
```

To run inference on the validation step of kittistep, follow the following steps

1. Create the inference_folder and the model_inference_outputs folder
```
mkdir .data/inference_folder/video_sequence/val
mkdir ./model_inference_outputs
```
2. Copy the validation set images to the inference_folder

```
!cp ./data/kitti-step/video_sequence/val/* ./data/inference_folder/video_sequence/val #copying the frames to the inference input folder 
```

3. Clean the model_inference_outputs folder in case there it's not empty
```
!rm -r ./model_inference_outputs/* #cleaning the previous inference outputs, if theu exist
```

4. run inference on the images (substitute `WEIGHTS_PATH` with the path to the weights you want to use for inference)

```
!bash ./tools/inference.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $WEIGHTS_PATH ./model_inference_outputs prova_da_sh
```

To evaluate the results run the following commands
1. To evaluate STQ
```
bash ./tools/evaluate_stq.sh ./model_inference_outputs
```
2. To evaluate VPQ
```
bash ./tools/evaluate_vpq.sh ./model_inference_outputs
```

## Running inference on your own frames.

To run inference on your frames follow the following procedure:

1. You should have already installed the environment by running the setup_videoknet_v2.sh

2. You should have a folder named "frames" in the main directory and put inside it the frames named and ordered according to the following convention: each image should have a filename beginning with 000002_. After that there must be an id identifying the number of the frame in format 60d. The filename should then with _leftImg8bit.png. So for example, if you have 10 frames, the first one should be named 000002_000001_leftImg8bit.png, the second one should be 000002_000002_leftImg8bit.png, the third 000002_000003_leftImg8bit.png and so on...

3. run the notebok `inference.ipynb`
