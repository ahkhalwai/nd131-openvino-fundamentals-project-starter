"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def pre_process(frame, net_input_shape):
    """
    Pre-process the image
    """
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

def draw_outputs(frame, network_output, prob_threshold, width, height):
    """
    Draw bounding boxes onto the frame.
    :param frame: frame from camera/video
    :param result: list contains the data comming from inference
    :return: person count and frame
    """
    pointer = 0
    probs = network_output[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > prob_threshold:
           pointer += 1
           box = net_output[0, 0, i, 3:]
           p1 = (int(box[0] * width), int(box[1] * height))
           p2 = (int(box[2] * width), int(box[3] * height))
           frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
    return frame, pointer

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    device = args.device
    cpu_extension = args.cpu_extension
            
    # Load the model through `infer_network` ###
    infer_network.load_model(model, cpu_extension, device)
    network_dimension = infer_network.get_input_shape()

    # Handle the input stream ###
    # Checks live feed
    if args.input == 'CAM':
        input_validated = 0

    # Checks input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)

    width = int(cap.get(3))
    height = int(cap.get(4))

    input_dimension = network_dimension['image_tensor']

    #iniatilize variables
    
    duration_prev = 0
    counter_total = 0
    dur = 0
    request_id=0
    
    report = 0
    counter = 0
    counter_prev = 0

    # Loop until stream is over #
    while cap.isOpened():
        # Read from the video capture #
        flag, frame = cap.read()
        if not flag:
            break

        # Pre-process the image as needed #
        processed_frame = pre_process(frame, input_dimension)
        
        # Start asynchronous inference for specified request #
        network_input = {'image_tensor': processed_frame,'image_info': processed_frame.shape[1:]}
        duration_report = None
        infer_network.exec_net(network_input, request_id)

        # Wait for the result #
        if infer_network.wait() == 0:

            # Get the results of the inference request #
            network_output = infer_network.get_output()

            # Extract any desired stats from the results #
            
            frame, pointer = draw_outputs(frame, network_output, prob_threshold, width, height)
            if pointer != counter:
                counter_prev = counter
                counter = pointer
                if dur >= 3:
                    duration_prev = dur
                    dur = 0
                else:
                    dur = duration_prev + dur
                    duration_prev = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > counter_prev:
                        counter_total += counter - counter_prev
                    elif dur == 3 and counter < counter_prev:
                        duration_report = int((duration_prev / 10.0) * 1000)

            # Calculate and send relevant information on #
            # current_count, total_count and duration to the MQTT server #
            # Topic "person": keys of "count" and "total" #
            # Topic "person/duration": key of "duration" #
            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)

        # Send the frame to the FFMPEG server #
        # Write an output image if `single_image_mode` #
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()