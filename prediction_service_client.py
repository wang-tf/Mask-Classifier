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
"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

import random
import logging

import grpc

import prediction_service_pb2
import prediction_service_pb2_grpc

import time
import cv2
import pickle


def guide_get_result(stub, image_path):
    request = prediction_service_pb2.PredictRequest()
    request.model_spec.name = "mask"
    request.model_spec.signature_name = "prediction"
    image = cv2.imread(image_path)
    #image = cv2.resize(image, None, fx=0.5, fy=0.5)
    request.inputs['input'] = pickle.dumps(image)
    start = time.time()
    response_future = stub.Predict.future(request)
    response = response_future.result()
    end = time.time()
    print('using time: ', end - start)
    print(pickle.loads(response.outputs['scores_output']))
    #print(pickle.loads(response.outputs['boxes_output']))


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    options = [('grpc.max_send_message_length', 1000 * 1024 * 1024),
               ('grpc.max_receive_message_length', 1000*1024*1024)]
    with grpc.insecure_channel('localhost:9001', options=options) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        print("-------------- GetResult --------------")
        guide_get_result(stub, './data/demo/test.jpg')
        guide_get_result(stub, './data/demo/demo1.jpg')
        guide_get_result(stub, './data/demo/demo1.jpg')
        guide_get_result(stub, './data/demo/demo1.jpg')


if __name__ == '__main__':
    logging.basicConfig()
    run()
