#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2023/05/07

@author: drumichiro
'''
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from deeplabv3 import DeepLabV3


# The labels of the PASCAL VOC dataset.
pascal_voc_labels = np.array([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])


# Python implementation of the color map function for the PASCAL VOC dataset.
# This function is based on https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae.
def get_pascal_voc_color_map():
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap


def illustrate(image_path):
    image_path = Path(str(image_path))

    device_name = "cpu"  # "cuda:0"
    model = DeepLabV3().to(device_name)
    model.load_state_dict(torch.load("models/deeplabv3.pth"))
    model.eval()

    print("Load an image. -> [%s]" % (image_path))
    image = Image.open(image_path)

    # Preprocessing.
    preprocess = transforms.Compose([
        transforms.Resize((257, 257)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(device_name)

    # Infer.
    y = model(x)

    # Create a mask of np.array.
    y = y.cpu().detach().numpy()
    # (1, 21, height, width) -> (height, width)
    segmentation = np.argmax(y[0], axis=0).astype(np.uint8)

    # Show detected labels.
    indices = np.unique(segmentation)
    print("Detected labels:", pascal_voc_labels[indices])

    # Plot the semantic segmentation predictions of 21 classes in each color.
    segmentation = Image.fromarray(segmentation).resize(image.size)
    cmap = get_pascal_voc_color_map()
    segmentation.putpalette(cmap)

    segmentation = Image.blend(image.convert('RGBA'),
                               segmentation.convert('RGBA'), alpha=0.6)

    # Added detected labels on the segmentation result.
    font_path = Path("C:/Windows/Fonts/NotoSansCJKjp-Black.otf")
    if font_path.exists():
        from PIL import ImageDraw, ImageFont
        font_size = 14
        draw = ImageDraw.Draw(segmentation)
        font = ImageFont.truetype(str(font_path), font_size)
        for ii, index in enumerate(indices):
            draw.text((5, ii * font_size), pascal_voc_labels[index], font=font,
                      fill=tuple(cmap[index]), stroke_width=2, stroke_fill="white")

    outpath = image_path.parent / Path("%s_segmentation.png" % (image_path.stem))
    print("Save a segmentation image. -> [%s]" % (outpath))
    segmentation.save(outpath)


def main():
    # import glob
    # for image_path in glob.glob("D:/VOC2012/JPEGImages/*.jpg"):
    #     illustrate(Path(image_path))
    illustrate("samples/family_usj_snw.jpg")
    illustrate("samples/segmentation_input1.jpg")
    illustrate("samples/segmentation_input2.jpg")


if __name__ == "__main__":
    main()
    print("Done.")
