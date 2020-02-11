# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide server."""

from concurrent import futures
import time
import math
import logging

import grpc

import prediction_service_pb2
import prediction_service_pb2_grpc

import os
import cv2
import pickle
import time
import numpy as np
from lib.SSH.SSH.test import detect
from func import SSH_init
from func import tf_init
from keras.applications import resnet50
import keras
#from keras.applications.resnet50 import ResNet50
#from deploy import classify
import tensorflow as tf
import caffe
import argparse

csize = 224

def classify(model, img_path='', img_arr=None):
    img_arr = cv2.resize(img_arr, (csize, csize), interpolation=cv2.INTER_NEAREST)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = resnet50.preprocess_input(img_arr)
    match = model.predict(img_arr)
    return match[0][0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9001, help='request network port')
    parser.add_argument('--per_process_gpu_memory_fraction', type=float, default=0.5, help='gpu used memory')
    parser.add_argument('--face_thresh', type=float, default=0.5, help='gpu used memory')
    args = parser.parse_args()
    return args

args = get_args()

def get_result(net, classify_model, image, face_thresh=0.5, max_size=0):
    if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
        start_time = time.time()
        bboxs, timer, logs = detect(net, im=image)
        print("FaceDetect Using time: ", (time.time() - start_time))
    img_cp = image.copy()

    len_line = int(img_cp.shape[1] / 5)
    pad_percent = int(img_cp.shape[1] / 2)
    x = int(img_cp.shape[1] / 25)
    y = int(img_cp.shape[0] / 25)
    pad_x = int(img_cp.shape[1] / 50)
    pad_y = int(img_cp.shape[0] / 25)
    pad_text = 5
    font_scale = (img_cp.shape[0] * img_cp.shape[1]) / (750 * 750)
    font_scale = max(font_scale, 0.25)
    font_scale = min(font_scale, 0.75)

    font_thickness = 1
    if max(img_cp.shape[0], img_cp.shape[1]) > 750: font_thickness = 2

    if bboxs.shape[0] == 0: return [], logs

    bboxs = bboxs[np.where(bboxs[:, -1] > face_thresh)[0]]
    bboxs = bboxs.astype(int)

    results = np.zeros((bboxs.shape[0], 5), dtype=float)
    results[:, :4] = bboxs[:, :4]
    for index, bbox in enumerate(bboxs):
        img_bbox = image[bbox[1]:bbox[3], bbox[0]:bbox[2], [2, 1, 0]]

        if img_bbox.shape[0] * img_bbox.shape[1] < max_size:
            continue

        prob = classify(classify_model, img_arr=img_bbox)
        results[index, 4] = prob

    return results, logs


class PredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        self.net = SSH_init()
        self.sess = tf_init(args.per_process_gpu_memory_fraction)
        self.classify_model = keras.models.load_model('./model/resnet50.h5')
        self.graph = tf.get_default_graph()

    def Predict(self, request, context):
        data = request.inputs['input']
        image = pickle.loads(data)
        logging.info('Input image shape: {}'.format(image.shape))
        start_time = time.time()
        with self.sess.as_default():
            with self.graph.as_default():
                results, logs = get_result(self.net, self.classify_model, image, face_thresh=args.face_thresh)
        end_time = time.time()
        response = prediction_service_pb2.PredictResponse()
        if results != []:
            response.outputs['boxes_output'] = pickle.dumps(results[:, :4])
            response.outputs['scores_output'] = pickle.dumps(results[:, 4:])
        else:
            response.outputs['boxes_output'] = pickle.dumps([])
            response.outputs['scores_output'] = pickle.dumps([])
        #response.log = 'detect using time: {}'.format(end_time-start_time)+'\n'+'\n'.join(logs)
        
        return response


def serve():
    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
               ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServiceServicer(), server)
    server.add_insecure_port('[::]:{}'.format(args.port))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    serve()
