# Video Panoptic Segmentation using Pix2Seq-D model in PyTorch

[Pix2Seq-D](https://arxiv.org/pdf/2210.06366.pdf) is a discrete denoising diffusion model for general purpose conditional discrete mask generation (such as panoptic masks). It is an adaptation of the approach from [Pix2Seq](https://arxiv.org/abs/2109.10852) combined with the discrete diffusion approach from [Bit Diffusion](https://arxiv.org/abs/2208.04202). This repository contains a PyTorch implementation of the model and training code for the Video Panoptic Segmentation task for pretraining on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and training on [Kitti-STEP](https://huggingface.co/LXT/VideoK-Net/tree/main).

The code is based on the Bit-Diffusion implementation by lucidrains available [here](https://github.com/lucidrains/bit-diffusion)

## Setting up the environment

Use conda or mamba (suggested) to create the environment from the environment.yml file

```
mamba env create -f env\environment.yml
```

## Pretraining on Cityscapes

To run the pretraining step on Cityscapes, you first need to download raw images and panoptic masks from the official command-line tool.

```
pip install cityscapesscripts
```

Then you can download the dataset with the following command (also available in the ```setup.sh``` script). It will prompt you for the download credentials. If you don't have an account, you can register [here](https://www.cityscapes-dataset.com/login/).

```
csDownload -d ./data/Cityscapes leftImg8bit_trainvaltest.zip gtFinePanopticParts_trainval.zip
```

You're ready to run the command

```
python pretrain.py
```

The command will save a checkpoint every 500 iterations in the ```results/pretrain``` folder. You can also download our pretrained checkpoints from [here](https://drive.google.com/drive/folders/1fN1ub4bFCrfy_JurUHuRP9cduIQzZpR2?usp=sharing).

## Training on Kitti-STEP

To run the training on KittiSTEP starting from a pretrained checkpoint on Cityscapes, copy your latest checkpoint to the ```results/train``` folder and name it ```model-0.pt```. The training procedure will automatically load the checkpoint and start training on KittiSTEP.

Simply run the command below. The code will automatically download the Kitti-STEP dataset from [here](https://huggingface.co/LXT/VideoK-Net/tree/main) if it's not present in the ```data``` folder, and start the training afterwards.

```
python train.py
```

## Inference

If you want to just use the model, our pretrained checkpoints are available here: [https://drive.google.com/drive/folders/1fN1ub4bFCrfy_JurUHuRP9cduIQzZpR2?usp=sharing](https://drive.google.com/drive/folders/1fN1ub4bFCrfy_JurUHuRP9cduIQzZpR2?usp=sharing)

Place the checkpoint in the ```results/train``` folder and name it ```model-latest.pt```.  
Place your data in the ```data/input``` folder and run the following command:

```
python inference.py
```

You can find the output in the ```data/output``` folder.

If you want to customize your input or output folders or your checkpoint, you can use the following command:

```
python inference.py --input_folder ./data/input --output_folder ./data/output --checkpoint ./results/train/model-latest.pt
```

See the full list of options with:

```
python inference.py -h
```

## Visualize the results

To visualize the results, you can use the ```visualize.ipynb``` notebook. It will show the input sequence, the predicted panoptic segmentation, semantic segmentation, and instance segmentation masks.

The results will be saved in the ```outputs/results``` folder and shown in the notebook.

## Results

Due to the lack of computing resources, we were able to run the pretraining and training procedures only for a little time (a few hours each on a single GPU). The results are not comparable to SOTA, but they show that the model is able to learn.

Below you can find a sequence from the Kitti-STEP validation set that shows the input sequence, the predicted instance segmentation, semantic segmentation, panoptic segmentation masks, and an overlay of the panoptic mask on the input sequence.

![Original Video](imgs/val/result_video.gif)

![Result Test Instance pix2seq](imgs/val/result_instance.gif)

![Result Test Semantic pix2seq](imgs/val/result_semantic.gif)

![Result Test Panoptic pix2seq](imgs/val/result_panoptic.gif)

![Result Test Video Overlay pix2seq](imgs/val/result_video_overlay.gif)