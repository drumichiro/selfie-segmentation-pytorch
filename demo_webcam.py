#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2023/05/03

@author: drumichiro
'''
import numpy as np
import cv2
import torch
from selfie_segmentation import SelfieSegmentation
import time


def put_text(image, text, y):
    cv2.putText(image, text=text,
                org=(0, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, text=text,
                org=(0, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


def show(model_path, mask_size, show_frames, device_name):
    model = SelfieSegmentation(*mask_size).to(device_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device_id = 0
    # device_id = "samples/family_usj_snw.jpg"
    capture = cv2.VideoCapture(device_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    resolution = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera resolution:", resolution)

    green_screen = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    inference_time = 0.0
    processing_time = 0.0
    inference_fps = 0.0
    processing_fps = 0.0
    processing_counter = 0
    perf_updates = 60

    while True:
        ret, frame = capture.read()
        if ret:
            processing_start = time.perf_counter()

            # Shrink.
            image = cv2.resize(frame, dsize=mask_size)

            # (height, width, 3) -> (1, 3, height, width)
            x = image.transpose([2, 0, 1])[np.newaxis, ...]

            # Normalize after conversion from np.array to torch.tensor.
            x = torch.from_numpy(x.astype(np.float32)).to(device_name) / 255.0

            # Infer.
            inference_start = time.perf_counter()
            y = model(x)
            inference_time += time.perf_counter() - inference_start

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

            # Expand.
            mask = cv2.resize(mask, dsize=resolution)

            # Apply smoothing.
            mask = cv2.medianBlur(mask, 5)

            # Convert a mask to a green screen.
            green_screen[..., 1] = 255 - mask

            # Overlay.
            frame = cv2.addWeighted(frame, 1.0, green_screen, 1.0, 0.0)

            processing_time += time.perf_counter() - processing_start

            # Update performance measurement.
            processing_counter += 1
            if 0 == (processing_counter % perf_updates):
                inference_fps = perf_updates / inference_time
                processing_fps = perf_updates / processing_time
                inference_time = 0.0
                processing_time = 0.0

            put_text(frame, "FPS", 25)
            put_text(frame, " Inference:%.1fHz" % (inference_fps), 50)
            put_text(frame, " Processing:%.1fHz" % (processing_fps), 75)

            cv2.imshow(model_path, frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or show_frames <= processing_counter:
                break

    capture.release()
    cv2.destroyAllWindows()

    print("Model:[%s](%s) -> Inference:%.1fHz Processing:%.1fHz" %
          (model_path, device_name, inference_fps, processing_fps))


def main():
    show("models/selfie_segmentation.pth", (256, 256), 300, "cpu")
    show("models/selfie_segmentation.pth", (256, 256), 300, "cuda:0")
    show("models/selfie_segmentation_landscape.pth", (256, 144), 300, "cpu")
    show("models/selfie_segmentation_landscape.pth", (256, 144), 300, "cuda:0")


if __name__ == "__main__":
    main()
    print("Done.")
