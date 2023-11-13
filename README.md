# Masked-LFW-Dataset-Mask-Parsing
This repository contains code for applying our face parsing technology to the Masked LFW Dataset. Our algorithm accurately identifies masked regions and sets their pixel values to 0, effectively segmenting the face from the mask. It is an essential step towards advanced facial recognition tasks in masked images.

<hr>

## Masked-LFW-Dataset
[MLFW: A Database for Face Recognition on Masked Faces](https://arxiv.org/abs/2109.05804)

, Chengrui Wang and Han Fang and Yaoyao Zhong and Weihong Deng.

You can download this dataset in [here](http://whdeng.cn/mlfw/?reload=true).

![1](https://github.com/Seungeun-Han/Masked-LFW-Dataset-Mask-Parsing/assets/101082685/cef0ac6a-e9fb-45d9-8a32-ef38849aae84)

<hr>

## Face_Parsing

We utilize our own Face Parsing Model, [This](https://github.com/Seungeun-Han/SCANet_Real-Time_Face_Parsing_Using_Spatial_and_Channel_Attention).

### Setting
- Input Size: 112 X 112

### Train_Dataset
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- CelebAMask-HQ_MaskRendering Dataset (Our Own)

<img width="405" alt="2" src="https://github.com/Seungeun-Han/Masked-LFW-Dataset-Mask-Parsing/assets/101082685/aa4cd948-8a67-4a07-bc6c-28763e97b998">

- Korean Mask Dataset (Our Own)

<hr>

## Results

![2](https://github.com/Seungeun-Han/Masked-LFW-Dataset-Mask-Parsing/assets/101082685/e3f221fd-6c2c-41c5-bd76-6d609cad6877)

<center><img src="https://github.com/Seungeun-Han/Masked-LFW-Dataset-Mask-Parsing/assets/101082685/e3f221fd-6c2c-41c5-bd76-6d609cad6877" width="80%" height="80%"></center>

This image illustrates the results of a face parsing technology applied to photographs of people wearing masks. Each row contains three images:

1. The left column shows the original face photos with masks on.

2. The middle column displays the detected mask regions, outlined in black against a white background. This binary mask representation is used for segmenting specific parts of the face.

3. The right column presents the outcome after the mask region has been removed (or set to 0), making it appear as if the mask is not present on the original photo.

These images demonstrate the application of facial parsing technology in computer vision, capable of accurately identifying and eliminating the portions covered by masks. Such technology is particularly crucial in fields like security systems, facial recognition software, or medical image analysis.

<hr>

### Copyrights
Copyright (c) 한승은. All rights reserved.

hse@etri.re.kr






