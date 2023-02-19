# MediaPipe Selfie Segmentation implemented in PyTorch

## INTRODUCTION

This is a PyTorch implementation of [MediaPipe Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation.html). You can see the network structure through the PyTorch scripts that are helpful to grasp it easily, I believe.

| Input Image | Mask Image   | Masked Image |
|-------------|--------------|--------------|
|![input_image](samples/family_usj_snw.jpg) | ![mask_image](samples/family_usj_snw_mask.png) | ![masked_image](samples/family_usj_snw_selfie.png) |

## HOW TO TRY

`python demo_static_image.py`


## REFERENCE
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
