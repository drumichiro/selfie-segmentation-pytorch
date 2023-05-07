# MediaPipe Image Segmentation implemented in PyTorch

## INTRODUCTION

This is a PyTorch implementation of [MediaPipe Image Segmentation](https://developers.google.com/mediapipe/solutions/vision/image_segmenter/). You can see the network structure through the PyTorch scripts that are helpful to grasp it easily, I believe.

## SAMPLES

### Selfie Segmentation
| Input Image | Mask Image   | Masked Image |
|-------------|--------------|--------------|
|![input_image](samples/family_usj_snw.jpg) | ![mask_image](samples/family_usj_snw_mask.png) | ![masked_image](samples/family_usj_snw_selfie.png) |
|![input_image](samples/family_usj_snw_landscape.jpg) | ![mask_image](samples/family_usj_snw_landscape_mask.png) | ![masked_image](samples/family_usj_snw_landscape_selfie.png) |

### Image Segmentation
| Input Image | Segmentation Image |
|-------------|--------------------|
|![input_image](samples/family_usj_snw.jpg) | ![masked_image](samples/family_usj_snw_segmentation.png) |
|![input_image](samples/segmentation_input1.jpg) | ![masked_image](samples/segmentation_input1_segmentation.png) |
|![input_image](samples/segmentation_input2.jpg) | ![masked_image](samples/segmentation_input2_segmentation.png) |

## HOW TO TRY

### Selfie Segmentation
* `$ python demo_selfie_segmentation.py`

### Image Segmentation
* `$ python demo_image_segmentation.py`

### Performance Measurement
* `$ python demo_webcam.py`

## REFERENCE
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
