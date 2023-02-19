#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2023/02/18

@author: drumichiro
'''
import numpy as np
import cv2
import torch
from selfie_segmentation import SelfieSegmentation


def main():
    model = SelfieSegmentation()
    model.load_state_dict(torch.load("models/selfie_segmentation.pth"))
    model.eval()

    basename = "family_usj_snw"

    path = "samples/%s.jpg" % (basename)
    print("Load an image. -> [%s]" % (path))
    image = cv2.imread(path)

    # (256, 256, 3) -> (1, 3, 256, 256)
    x = image.transpose([2, 0, 1])[np.newaxis, ...]

    # Normalize after conversion from np.array to torch.tensor.
    x = torch.from_numpy(x.astype(np.float32)) / 255.0

    # Infer.
    y = model(x)

    # (1, 1, 256, 256) -> (256, 256)
    y = y[0, 0, ...]

    # Create a mask of np.array.
    y = y.cpu().detach().numpy()
    mask = (y.copy() * 255).astype(np.uint8)

    if False:
        # Apply a threshold to create a hard mask.
        threshold = 0.3
        mask[y >= threshold] = 255
        mask[y < threshold] = 0

    path = "samples/%s_mask.png" % (basename)
    print("Save a mask image. -> [%s]" % (path))
    cv2.imwrite(path, mask)

    # Green Screen as a background.
    selfie = np.zeros(image.shape, dtype=np.uint8)
    selfie[..., 1] = 255 - mask
    # These transposes are just to make the image broadcastable to multiply y.
    image = (image.transpose([2, 0, 1]) * y).astype(np.uint8).transpose([1, 2, 0])
    selfie = cv2.addWeighted(image, 1.0, selfie, 1.0, 0.0)

    path = "samples/%s_selfie.png" % (basename)
    print("Save a selfie image. -> [%s]" % (path))
    cv2.imwrite(path, selfie)


if __name__ == "__main__":
    main()
    print("Done.")
