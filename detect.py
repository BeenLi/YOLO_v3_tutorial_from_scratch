from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

# Variable(deprecated):Variable(tensor) still work,but return tensors instead of Variables
# 以前tensor是不能参与反向传播的,需要进行variable = Variable(tensor, requires_grad=True)
# var.data is the same thing as `tensor.data`

import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguments to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
# CUDA = torch.cuda.is_available()
CUDA = False

num_classes = 80
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso  # input resolution
inp_dim = int(model.net_info["height"])
input_size = (inp_dim, inp_dim)       # 由于输入默认是正方形,所以width和height相等
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU available, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)] # assume images is a directory
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det): # the destination of storing images after detection
    os.makedirs(args.det)

load_batch = time.time()
loaded_imgs = [cv2.imread(x) for x in imlist] # imlist is a list containing the path of images

im_batches = list(map(prep_image, loaded_imgs, [input_size[0] for x in range(len(imlist))])) # all the pictures in a directory
# stops when the shortest iterable is exhausted

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs] # 细节;cv2.imread读出来时时[height,width,channel]
# 但是cv2对坐标的计算是按(x, y) = (width, height)
im_dim_list = torch.FloatTensor(im_dim_list) # why repeat?

leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover # the leftover need a batch
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_batches))]), 0) for i in range(num_batches)]

write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()

start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    """
    we iterate over the batches, generate the prediction.
    and concatenate the prediction tensors(shape = D * 8, the output of `write_results` function.
    of all the images we have to perform detection upon
    """
    # load the image
    start = time.time() # detection time
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad(): # deserve thinking
        # 在该语句下,不记录梯度
        prediction = model(batch, CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

    end = time.time()

    if type(prediction) == int: # meaning there is no detection because we return 0

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size  # transform the attribute from index in batch to index in imlist

    if not write:  # If we haven't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    # output the prediction result of one batch
    for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]): # image path
        im_id = i * batch_size + im_num # the # picture
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()
        # The line torch.cuda.synchronize makes sure that CUDA kernel is synchronized with the CPU
try:
    output
except NameError:
    print("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long()) # long: float32-->int64
# Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTens


scaling_factor = torch.min(torch.cat(((input_size[0] / im_dim_list[:,1]).unsqueeze(1),
                                      (input_size[1]/ im_dim_list[:,0]).unsqueeze(1)),1), 1)[0].view(-1,1)
# (value, index) = torch.min(tensor, 1)
# inp_dim = args.reso
# (h - new_h)//2 --> (h - image_h*scale)//2
output[:, [1, 3]] -= (input_size[0] - scaling_factor * im_dim_list[:, 0].view(-1, 1)) // 2 # top-left x and bottom-right x
output[:, [2, 4]] -= (input_size[1] - scaling_factor * im_dim_list[:, 1].view(-1, 1)) // 2

output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results): # results is the loaded original image from cv2, x contain the opposite coordinate
    c1 = tuple(x[1:3].int()) # c1 is the top-left (x,y)
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])] # find the corresponding image
    cls = int(x[-1]) # find the class index
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 2, lineType=cv2.LINE_AA) # 2: thickness
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0] # font_scale, font_thickness (width, height)
    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4 # c2 at the right-top of original rectangle's top-left corner
    # (width:x, height:y)
    cv2.rectangle(img, c1, c2, color, -1, lineType=cv2.LINE_AA) # -1 fill the rectangle
    cv2.putText(img, label, (c1[0], c1[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1, bottomLeftOrigin=True);
    # 将字台高，防止字出边界
    # org Bottom-left corner of the text string in the image
    return img


list(map(lambda x: write(x, loaded_imgs), output))
# in windows the path separator is "\\" or "\" , but in linux it's "/"


def get_det_name(x):
    return "{}\\det_{}".format(osp.join(osp.realpath('.'), args.det), x.split("\\")[-1])


det_names = pd.Series(imlist).apply(get_det_name)

list(map(cv2.imwrite, det_names, loaded_imgs))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()
