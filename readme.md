# Diffusion Models vs Learnable Kernels for Video Panoptic Segmentation
## Ceraolo Martorella Minini DLAV2023 Final Project
[Click here for the short video presentation](https://youtu.be/lESWO-GZMV4)

## Introduction - What’s Video Panoptic Segmentation?
 Panoptic segmentation is an advanced computer vision task that combines instance segmentation and semantic segmentation to provide a comprehensive understanding of the visual scene in a video. It aims to label every pixel in the video with a class label and an instance ID, thereby differentiating between different objects and identifying their boundaries. This task involves simultaneously segmenting and tracking "things" (individual objects) and "stuff" (background regions). Handling temporal coherence and tracking objects across frames are important considerations to achieve accurate and consistent segmentation results. It plays a crucial role in various applications, including autonomous driving, video surveillance, augmented reality, and robotics, enabling machines to perceive and interpret the environment more comprehensively.


## Approach - two different architectures
The two methods that we studied and worked on are Pix2Seq-D and Video K-Net.

### Pix2Seq-D - Diffusion Models for VPS
Pix2Seq a generalist approach to segmentation. The authors present a formulation of panoptic segmentation as a discrete data generation problem, without relying on specific task-related biases. They use a diffusion model based on analog bits, which is a probabilistic generative model, to represent panoptic masks. The model has a simple and generic architecture and loss function, making it applicable to a wide range of panoptic segmentation tasks. By leveraging the power of the diffusion model and incorporating temporal information, the proposed approach offers a promising alternative for panoptic segmentation tasks without the need for task-specific architectures or loss functions.


### Video K-Net - Video Panoptic Segmentation with Learnable Kernels
Video K-Net is a framework for fully end-to-end video panoptic segmentation. The method builds upon K-Net, a technique that unifies image segmentation using learnable kernels. The authors observe that the learnable kernels from K-Net, which encode object appearances and contexts, can naturally associate identical instances across different frames in a video. This observation motivates the development of Video K-Net, which leverages kernel-based appearance modeling and cross-temporal kernel interaction to perform video panoptic segmentation.


## History - our project
We believe that the idea behind Pix2Seq-D is incredibly fascinating and powerful, and that’s why we were very keen on working with it, notwithstanding the issues we had. We spent several weeks trying to run the Tensorflow code provided by Google Research, but we encountered numerous issues that prevented us from using their code (see the section [#issues](#side-notes:-issues-we-had) for more details). We tried a huge amount of solutions, different setups, several GPUs and GPU providers, and so on, without success. So more recently, we decided to embark on an ambitious mission: rewriting the Pix2Seq-D codebase in PyTorch. Fortunately, individual contributors on Github already converted some sub-parts of the project (e.g. [Bit-Diffusion by lucidrains](https://github.com/lucidrains/bit-diffusion)). After some heavy work, we actually managed to have a draft of the full project. It is now running the training for the very first time, so we don’t expect perfect results yet. We plan on pursuing and completing this project also after the milestone deadline.
 In parallel, since we knew about the uncertainty of such a challenge, we also setup and run the training of another architecture, Video K-Net, so that we also have outputs to show, and a baseline performance to compare the results of our main contribution.

## Contribution overview
Our main contribution is within the Pix2Seq-D architecture. The contributions are three: the re-implementation of the whole codebase in Pytorch, and the adaptation of the architecture to the task of Video Panoptic Segmentation. We also ran the training of another architecture, Video K-net, in order to have a solid benchmark, with same datasets for pre-training and training.

- First of all, we believe that our re-implementation in Pytorch can help the scientific community, ensuring more interoperability, clarity, and extending the audience that can build solutions on top of Pix2Seq-D. Diffusion models are very recent and proved to be very powerful, so the more researchers can have access to resources and codebases, the faster innnovations will come. It also improves the reproducibility of Pix2Seq-D, since their tensorflow codebase is complex to use and the setup is unclear.

- Secondly, we adapted the codebase to the task of Video Panoptic Segmentation. The authors of the paper used it for Image Panoptic Segmentation and Video Segmentation, but not for Video Panoptic Segmentation. We believe that this is a very interesting task, and that the diffusion model can be very powerful for it.

Finally, we ran another architecture, Video K-net, in order to have a solid benchmark, with same pre-training and training, and become familiar with the panoptic mask generation process.


## Experimental setup
With both architectures, we kept consistency of the training procedure. We used the following experimental setup:
- Backbone: ResNet50 pre-trained on ImageNet for Video K-net, ResNet50 FPN pre-trained on COCO from Torchvision's [FasterRCNN_ResNet50_FPN_V2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2) for Pix2Seq-D
- pre-training on Cityscapes
- training on KITTI-STEP
See the following section for more information on datasets.

In the Video K-net case, we used the checkpoints provided by the authors of the paper for the pre-training on Cityscapes. In the Pix2Seq-D case, we ran the pre-training on Cityscapes ourselves from scratch, and then also the training on KITTI-STEP for Video Panoptic Segmentation.

We both did qualitative and quantitative evaluations. The qualitative comprised creating a script to visualize a gif video to see the colored panoptic masks and do visual inspection, together with comparison with the ground truth. The quantitative comprised the following measures:
- Segmentation and tracking quality (STQ). STQ takes into account both segmentation quality (how accurate the object boundaries are) and tracking quality (how well the objects are tracked across consecutive frames).
- Association Quality (AQ). AQ assesses the quality of the association between the identified segments and the ground truth objects in video panoptic segmentation. It measures how well the model assigns the correct identity to each segment, indicating the accuracy of the instance-level association.

Finally, since the training processes were heavy, we used powerful GPUs. More specifically, we used:
- an RTX3090 for Video K-net
- an A100 80GB for Pix2Seq-D

## Data
The dataset used for pre-training is Cityscapes. Cityscapes dataset is a high-resolution road-scene dataset which contains 19 classes. (8 thing classes and 11 stuff classes). It contains 2975 images for training, 500 images for validation and 1525 images for testing.

Our pre-training for Pix2Seq-D was done with the images in .png format. The dataset can be downloaded with the [official scripts](https://github.com/mcordts/cityscapesScripts)


The dataset used for training is KITTI-STEP. 
It can be obtained from this [Huggingface link](https://huggingface.co/LXT/VideoK-Net/tree/main), thanks to the authors of Video K-net. We obtained the dataset from there both for training Pix2Seq-D and Video K-net. This dataset contains videos recorded by a camera mounted on a driving car. KITTI-STEP has 21 and 29 sequences for training and testing, respectively. The training sequences are further split into training set (12 sequences) and validation set (9 sequences). Since the sequence length sometimes reaches over 1000 frames, long-term tracking is required. Prior research has shown that with increasing sequence length, the tracking difficulty increases too.
The labels of the dataset, which are the panoptic masks, are provided in .png format. More specifically, the groundtruth panoptic map is encoded as follows in PNG format:

```
R = semantic_id
G = instance_id // 256
B = instance % 256
```

The following is the label distribution in KITTI-STEP (image taken from "STEP: Segmenting and Tracking Every Pixel" paper):


<img src="images_readme/label_distribution_kitti.png" alt="Label Distribution" width="350">


## Results
Qualitative and Quantitative results of your experiments. 

### Qualitative evaluation

### Video K-net (on train set)
The following are the masks obtained with Video K-net, pretrained on Cityscapes and trained on KITTI-STEP.

Inference is done on a sequence of the train set of KITTI-STEP. The first video is the input, the second is the predicted instance segmentation mask, the third is the predicted semantic segmentation mask, the fourth is the predicted panoptic segmentation mask, and the fifth is the overlay of the predicted panoptic mask on the input video.

![Original Video](output_gifs_videoknet/train_set/result_video.gif)

![Result Train Instance Videoknet](output_gifs_videoknet/train_set/result_instance.gif)

![Result Train Semantic Videoknet](output_gifs_videoknet/train_set/result_semantic.gif)


![Result Train Panoptic Videoknet](output_gifs_videoknet/train_set/result_panoptic.gif)

![Result Train Video Overlay Videoknet](output_gifs_videoknet/train_set/result_video_overlay.gif)




### Video-k-net (on val set)
Here, inference is done on a sequence of the validation set of KITTI-STEP. The model used is the same as above.
![Original Video](output_gifs_videoknet/val_set/result_video.gif)

![Result Test Instance Videoknet](output_gifs_videoknet/val_set/result_instance.gif)

![Result Test Semantic Videoknet](output_gifs_videoknet/val_set/result_semantic.gif)


![Result Test Panoptic Videoknet](output_gifs_videoknet/val_set/result_panoptic.gif)

![Result Train Video Overlay Videoknet](output_gifs_videoknet/val_set/result_video_overlay.gif)

### Pix2Seq-D (on train set)
The following are the masks obtained with our implementation of Pix2Seq-D, pretrained on Cityscapes and trained on KITTI-STEP.

Inference is done on a sequence of the train set of KITTI-STEP. The first video is the input, the second is the predicted instance segmentation mask, the third is the predicted semantic segmentation mask, the fourth is the predicted panoptic segmentation mask, and the fifth is the overlay of the predicted panoptic mask on the input video.

The results of Pix2Seq-D are definetly improvable, and worse than the ones given by Video K-net. This is probably due to the fact that we did not have enough time to train the model properly for enough epochs, not enough computational power to tune the hyperparameters. However, we believe that the results are still good, and that the model is able to learn. This brought us to believe that, given more time and computational power, we could obtain better results.

![Original Video](output_gifs_pix2seq/train_set/result_video.gif)

![Result Train Instance pix2seq](output_gifs_pix2seq/train_set/result_instance.gif)

![Result Train Semantic pix2seq](output_gifs_pix2seq/train_set/result_semantic.gif)

![Result Train Panoptic pix2seq](output_gifs_pix2seq/train_set/result_panoptic.gif)

![Result Train Video Overlay pix2seq](output_gifs_pix2seq/train_set/result_video_overlay.gif)

### Pix2Seq-D (on test set)
Here, inference is done on a sequence of the validation set of KITTI-STEP. The model used is the same as above.


![Original Video](output_gifs_pix2seq/val_set/result_video.gif)

![Result Test Instance pix2seq](output_gifs_pix2seq/val_set/result_instance.gif)

![Result Test Semantic pix2seq](output_gifs_pix2seq/val_set/result_semantic.gif)

![Result Test Panoptic pix2seq](output_gifs_pix2seq/val_set/result_panoptic.gif)

![Result Test Video Overlay pix2seq](output_gifs_pix2seq/val_set/result_video_overlay.gif)


### Quantitative evaluation 

The following are the quantitative results obtained with our implementation of Pix2Seq-D, pretrained on Cityscapes and trained on KITTI-STEP.

|                             | STQ     | AQ     |
|-----------------------------|---------|--------|
| Our Results (Video K-net)    | 0.66    | 0.69   |
| Authors' Results (Video K-net)| 0.71    | 0.70   |

We were not yet able to gather quantitative metrics on Pix2Seq-D. We would need to train the model for more epochs, on a larger dataset, and have more GPU time available to run inference on the validation set.


## Code and replication
In the subfolders of this repository, you can find the code for the two models we used, both our implementation of Pix2Seq-D and the code we used to run Video K-net. You can also find the procedures to run the code, and the links to download the checkpoints. 

## Next steps
We are confident that Pix2Seq-D can be improved and has large performance potential, and we have some ideas on how to do it. First of all, encouraged by the loss decrasing during the few epochs of training that we ran (see [Pix2Seq-D's README](./Pix2Seq-D/README.md)), we believe that the model can be trained for more epochs, and that this will lead to better results. Moreover, the model can be further improved by training it on larger datasets, both in pretraining and in training phase. Finally, we would also want to consider to do some changes to the architecture. For instance, we have the following idea in mind. The solution they propose to do Video Panoptic Segmentation is by using the Diffusion process, and when predicting the frame at time t, they guide the diffusion by conditioning on the previous panoptic mask. Our idea is the following: instead of conditioning only on the previous mask, we plan on finding a way to compute the difference between the current frame and the previous one, and condition on such difference, together with the previous mask. The idea is that, given two frames, it is very likely that there will not be extreme changes, but mainly instances that moved by some pictures. The difference between the frames will highlight what changed, and hence may make the diffusion process to find the mask faster, and more accurate. Since we had to re-write the whole codebase, we were not able to implement such configuration yet.


## Side notes: issues we had
We believe it can be useful to share with the EPFL community the issues that we encountered while running Tensorflow code, both on SCITAS and on external GPU providers.

- The computational load of the training was very high and thus we cannot run the code on our machines, as memory got filled almost immediately. 

- Thus, we had been trying to run the code on the Izar cluster but we encountered lots of issues there as well. Mainly, there were issues with setting up the environment and having the right packages installed and working together without conflicts. Eventually, we did manage to install the right versions of the packages, but still there was an issue with the path of an ssl certificate required by tensorflow. 

- To solve this issue, we set up Docker on the cluster but we could not build the docker image, since the instructions and line commands found on the SCITAS documentation were outdated. 
- We tried to solve this issue by contacting the technical desk, but unfortunately they have not been able to find an effective solution and they refused to add a symlink on the cluster, which would have solved our issue. 
- The technical desk has contacted us again and they proposed some other solutions, but at that point it was too late to try them out because of the restricted reserved slots.

- In the meanwhile, we also tried alternatives to run the code. We tried to run it on cloud computing services such as Colab, Kaggle, Google Cloud, and Microsoft Azure, but each of them has its own problems. Colab does not offer enough GPU memory, Kaggle does but the training breaks because of an issue with a tensorflow library.

All this problems brought us to the conclusion that the best solution was to convert the codebase to Pytorch, which is more flexible and easier to run on different platforms. We also found the best solution to run the code on GPUs, which external pay-per-use GPUs, namely [runpod.io](https://www.runpod.io).

## References
```
@inproceedings{li2022videoknet,
  title={Video k-net: A simple, strong, and unified baseline for video segmentation},
  author={Li, Xiangtai and Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Cheng, Guangliang and Tong, Yunhai and Loy, Chen Change},
  booktitle={CVPR},
  year={2022}
}

@article{chen2022unified,
  title={A generalist framework for panoptic segmentation of images and videos},
  author={Chen, Ting and Li, Lala and Saxena, Saurabh and Hinton, Geoffrey and Fleet, David J.},
  journal={arXiv preprint arXiv:2210.06366},
  year={2022}
}
```
