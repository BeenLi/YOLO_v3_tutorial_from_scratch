from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *


def get_test_input():
    img = cv2.imread("dog-cycle-car.png") # img.size=816321(452*602*3)
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension(416,416,3)
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H * W * C -> C * H * W (3,416,416)
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    # noinspection PyUnresolvedReferences
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store all characters in a string, then using "\n" separates
    # to get line string list
    lines = [x for x in lines if len(x) > 0]  # get rid of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe(边缘)whitespaces

    block = {}  # create a dictionary
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()  # remove the brackets on both sides of the string
        else:
            key, value = line.split("=")  # split a string to a list ,i.e.['batch', '1']
            block[key.rstrip()] = value.lstrip()  # in case of the space on both sides of "="
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    :param blocks: (return from the parse_cfg function)
    :return: nn.ModuleList(is almost like a normal list containing nn.Module
    """
    # blocks[0] is the first layer[net] in yolov3.cfg
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()  # torch.nn.modules.container.ModuleList
    prev_filters = 3  # the number of filters in the previous layer,i.e.the depth of the filter this layer
    output_filters = []  # prepare for route layer

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()  # sequentially execute a number of nn.Module objects
        # for example the conv layer is "conv + BN + Leaky relu"
        # check the type of block
        # create a new module for the block
        # append to module_list

        # If it's a convolutional layer
        if x["type"] == "convolutional":
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2  # padding is a flag whether using pad
                # stride controls the output. if stride = 1, output the same size; if stride = 2, output half size
                # to ensure [n+2*p-f]//stride + 1 == n // stride
            else:
                pad = 0

            # Add the convolutional layer(in_channel, out_channel..)
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)  # name:str, module

            # Add the Batch Norm Layer
            # nn.BatchNorm2d(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
            # num_features，输入数据的通道数，归一化时需要的均值和方差是在每个通道(feature map)中计算的
            # eps，用来防止归一化时除以0
            # momentum，滑动平均的参数，用来计算running_mean和running_var
            # affine，是否进行仿射变换，即缩放操作
            # track_running_stats，是否记录训练阶段的均值和方差，即running_mean和running_var(跟踪不同batch的均值和方差,但仍然使用每个batch的均值和方差归一化)
            # running_mean = (1-momentum)*running_mean + momentum*mu ---> when inference, we don't calculate the mean and var
            # running_var = (1-momentum)*running_var + momentum*sigma

            # BN层有五个参数
            # weight，缩放操作的γ \gammaγ。
            # bias，缩放操作的β \betaβ。
            # running_mean，训练阶段统计的均值，测试阶段会用到。
            # running_var，训练阶段统计的方差，测试阶段会用到。
            # num_batches_tracked，训练阶段的batch的数目，如果没有指定momentum，则用它来计算running_mean和running_var。
            # 一般momentum默认值为0.1，所以这个属性暂时没用。
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)  # the number of filter is equal to the number of feature map
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif x["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # magnifier the input
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive annotation
            if start > 0:
                start = start - index  # unify the filter expression; if it is negative, it is an offset
                # but if it is positive, it is ground truth layers index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)  # dummy layer
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
                # after route layer the number of channels
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  # [(10,13),(16,30)...]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)  # self_created layer with attribute of anchors
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters  # prepare for the next layer
        output_filters.append(filters)

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()  # inherit father class initial process
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        """

        :param x: layer input
        :param CUDA: My Nvidia cuda toolkit == 11.0
        :return: output
        """
        modules = self.blocks[1:] # module attributes list, and each element is a dict
        outputs = {}  # We cache the outputs for the route layer

        write = 0 # indicate whether we have encountered the first detection or not
        # because we can't concatenate a non-empty tensor to a empty tensor
        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)  # directly pass into neural layer

            elif module_type == "route":
                # we assume there are only two layers being concatenated
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]  # acquire the prev layer output

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]  # feature map in one layer's output
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data       # only fetch the date from tensor object
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:  # if no collector has been initialized.
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1) # cat at the 1th dimension
                    # [batch_size, grid_size*grid_size*num_anchors, bbox_attrs]
                    # three different scale detection layer have the same bbox_attrs:(B(5+C))
            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"] # not remove the net layer, so add one

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model_ = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model_[0]  # what does this mean? model_ is [conv,bn,leaky]

                if batch_normalize:
                    bn = model_[1]

                    # Get the number of weights of Batch Norm Layer
                    # BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases(conv output feature map)
                    num_biases = conv.bias.numel()

                    # Load the conv biases
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


if __name__ == "__main__":
    model = Darknet(".\cfg\yolov3.cfg")
    inp = get_test_input()
    model.load_weights("yolov3.weights")
    pred = model.forward(inp, torch.cuda.is_available())
    write_results(pred, 0.9, 80)
    # filename = ".\cfg\yolov3.cfg"
    # blocks = parse_cfg(filename)
    # print(create_modules(blocks))
