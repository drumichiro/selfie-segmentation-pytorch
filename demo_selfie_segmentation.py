#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2023/02/18

@author: drumichiro
'''
from pathlib import Path
import numpy as np
import cv2
import torch
from selfie_segmentation import SelfieSegmentation


def illustrate(image_path, model_path, width, height):
    image_path = Path(str(image_path))
    model_path = Path(str(model_path))

    model = SelfieSegmentation(width, height)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Load an image. -> [%s]" % (image_path))
    image = cv2.imread(str(image_path))

    # I am not sure why, but the created mask looks better to me in BGR than in RGB.
    if True:
        # (height, width, 3) -> (1, 3, height, width)
        x = image.transpose([2, 0, 1])[np.newaxis, ...]
    else:
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose([2, 0, 1])[np.newaxis, ...]

    # Normalize after conversion from np.array to torch.tensor.
    x = torch.from_numpy(x.astype(np.float32)) / 255.0

    # Infer.
    y = model(x)

    # (1, 1, height, width) -> (height, width)
    y = y[0, 0, ...]

    # Create a mask of np.array.
    y = y.cpu().detach().numpy()
    mask = (y.copy() * 255).astype(np.uint8)

    if False:
        # Apply a threshold to create a hard mask.
        threshold = 0.3
        mask[y >= threshold] = 255
        mask[y < threshold] = 0

    outpath = image_path.parent / Path("%s_mask.png" % (image_path.stem))
    print("Save a mask image. -> [%s]" % (outpath))
    cv2.imwrite(str(outpath), mask)

    # Green Screen as a background.
    selfie = np.zeros(image.shape, dtype=np.uint8)
    selfie[..., 1] = 255 - mask
    # These transposes are just to make the image broadcastable to multiply y.
    image = (image.transpose([2, 0, 1]) * y).astype(np.uint8).transpose([1, 2, 0])
    selfie = cv2.addWeighted(image, 1.0, selfie, 1.0, 0.0)

    outpath = image_path.parent / Path("%s_selfie.png" % (image_path.stem))
    print("Save a selfie image. -> [%s]" % (outpath))
    cv2.imwrite(str(outpath), selfie)


def main():
    illustrate("samples/family_usj_snw.jpg",
               "models/selfie_segmentation.pth", width=256, height=256)
    illustrate("samples/family_usj_snw_landscape.jpg",
               "models/selfie_segmentation_landscape.pth", width=256, height=144)


if __name__ == "__main__":
    main()
    print("Done.")
