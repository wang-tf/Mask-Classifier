import os
import cv2
import time
import numpy as np
from lib.SSH.SSH.test import detect
from func import SSH_init


def demo_image(net, image_path, save_out=False, out_path='./data/images/output.jpg'):
    assert os.path.exists(image_path), image_path

    frame = cv2.imread(image_path)
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:

        # with videos we don't use SSH pyramid option to improve performance
        for i in range(10):
            start_time = time.time()
            bboxes, _, logs = detect(net, im=frame)

            print("FPS: ", 1.0 / (time.time() - start_time))

        if save_out:
            cv.imwrite(out_path, frame)

    return bboxes


def demo_detect(net, im):
    bboxes, _, logs = detect(net, im=im)
    return bboxes, _, logs


def main():
    net = SSH_init()

    #demo_video(net, visualize=True)
    
    # uncomment below to run demo on video
    #demo_video(net, './data/videmo/test1.mp4', save_out=False, visualize=False)
    demo_image(net, './data/demo/demo1.jpg', save_out=False)


if __name__ == "__main__":
    main()

