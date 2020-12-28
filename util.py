from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np) # device:cpu

    tensor_res = tensor.new(unique_tensor.shape) # device=`cuda:0`
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """

    :param prediction: (input of yolo layer)our output[batch_size, channel, height, width]
    :param inp_dim: input image dimension
    :param anchors:
    :param num_classes:
    :param CUDA:
    :return: 2-D tensor(ignore the batch size dimension)(each row corresponds to attributes of a bounding box).
    """

    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)  # scale size
    grid_size = inp_dim // stride  # detection map ?= prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    # [batch_size, bbox_attrs*num_anchors, grid_size, grid_size]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    # [batch_size, grid*grid, bbox_attrs*num_anchors]
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    # after transform, each row only has one box
    anchors = [(a[0] / stride, a[1] / stride) for a in
               anchors]  # scale;because the anchor is relative to the origin input

    # Sigmoid the  centre_X, centre_Y. and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)  # np.arrange(5)=[0,1,2,3,4]
    a, b = np.meshgrid(grid, grid)  # (x,y)---> x=a[x][y] and y=b[x][y]

    x_offset = torch.FloatTensor(a).view(-1, 1)  # change to column vector size:[grid*grid,1]
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # x_y_offset's size equals [1, grid*grid*num_anchors, 2]; Add broadcast
    # because unsqueeze(0) so there is an addition dimension with size of 1
    prediction[:, :, :2] += x_y_offset  # all the x and y
    # prediction[:,:,:2]'size equals [batch_size, grid*grid*num_anchors, 2]

    # log space transform height and the width
    # anchors = [(3, 2), (2, 4), (4, 3)] : three anchor boxes height and width
    anchors = torch.FloatTensor(anchors)  # the size of anchors is [num_anchors,2]

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)  # the size is [1, grid*grid*num_anchors, 2]
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # pw * exp(tw)

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride  # resize the detections map to the size of the input image

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    we must subject our output to objectness score threshold and Non-maximal suppression
    :param prediction: [batch_num, 108647, 85]
    :param confidence: objectness score threshold
    :param num_classes: 80 in our case
    :param nms_conf: the NMS Iou threshold
    :return:
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # float: if x = True --> 1; if x = false ---> 0
    # conf_mask shape is [batch_size, 10647,1]
    # prediction shape is [batch_size, 10647, 85]
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    # top-left corner (x,y) and right-bottom corner (x,y)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2) # top-left corner x
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    # indicates that we haven't initialized output,
    # a tensor we will use to collect true detections across the entire batch.
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        # size = [10647, 85]
        # confidence thresholding
        # NMS

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # (values, indices)
        max_conf = max_conf.float().unsqueeze(1) # must unsqueeze(1), otherwise the shape is [10647]
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1) # get rid of 80 class probabilities and instead inserting tow attributes
        # shape [10647, 7]
        # max_index and max_score

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        # return a tensor containing the indices of all non-zero elements of input
        try:
            # in case of the situation where we get no detections
            # non_zero_ind's shape is [num_non_zero, 1],after squeeze(), it become 1-D;
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # if image_pred_.shape[0] == 0:
        #     continue

        # Get the various classes detected in the image
        # in case of multiple true detections of the same class
        # get rid of the repeat class
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1) # size:[num_predict,7];the row that is not the class become zero
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind, :].view(-1, 7) # change the class_mask_ind

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            # (values, indices)
            image_pred_class = image_pred_class[conf_sort_index] # descending ordered by class probability
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    # it may be image_pre_class[[i],:]
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            # a.new()该Tensor的type和device都和原有Tensor一致，且无内容。
            # Repeat the batch_id for as many detections of the class cls in the image
            seq = (batch_ind, image_pred_class)

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


def letterbox_image(img, inp_dim):
    """resize image with unchanged aspect ratio using padding"""
    img_w, img_h = img.shape[1], img.shape[0] # img:[height, width, channel]
    w, h = inp_dim
    scale = min(w / img_w, h / img_h) # keep aspect ratio
    new_w = int(img_w * scale) # multiply the same number
    new_h = int(img_h * scale)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC) # cubic
    # 表示大小时用的是(width,height) --> 返回的是[height, width, channel]??
    canvas = np.full((h, w, 3), 128) # create a numpy array having shape of [width, height, c]

    # padding with (128,128,128) gray
    canvas[(h - new_h) // 2: (h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    OpenCV loads an image as an numpy array, with BGR as the order of the color channels
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy() # Return a copy of the array
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) # add a dimension having size of 1 in order to cat
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
