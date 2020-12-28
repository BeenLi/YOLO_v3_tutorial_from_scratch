import cv2
import numpy as np


def letterbox_image(img, inp_dim):
    """resize image with unchanged aspect ratio using padding"""
    img_w, img_h = img.shape[1], img.shape[0] # img:[height, width, channel]
    w, h = inp_dim
    scale = min(w / img_w, h / img_h) # keep aspect ratio
    new_w = int(img_w * scale) # multiply the same number
    new_h = int(img_h * scale)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC) # cubic
    cv2.imshow("unchanged aspect ratio", resized_image)
    # 表示大小时用的是(width,height) --> 返回的是[height, width, channel]??
    canvas = np.full((h, w, 3), 128) # create a numpy array having shape of [width, height, c]

    # padding with (128,128,128) gray
    canvas[(h - new_h) // 2: (h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

if __name__ == "__main__":
    filename = "imgs\\dog.jpg"
    dog = cv2.imread(filename)
    cv2.imshow("original_dog", dog)
    # shrink_dog = cv2.resize(dog, None, fx=0.5, fy=1, interpolation=cv2.INTER_CUBIC)
    # dog_keep_ratio_padding = np.uint8(letterbox_image(dog, (dog.shape[0] // 2, dog.shape[0] // 2))) # height/2
    # cv2.imshow("dog_keep_ratio_padding", dog_keep_ratio_padding)
    # cv2.imshow("shrink_dog", shrink_dog)
    c1 = (164, 108)
    c2 = (560, 447)
    color = (255, 0, 0) # red
    cv2.rectangle(dog, c1, c2, color, 2, lineType=cv2.LINE_AA)
    cv2.imshow("rectangle", dog)
    label = "bicycle"
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0] # font_scale, font_thickness (width, height)
    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4 # c2 at the right-top of original rectangle's top-left corner
    # (width:x, height:y)
    cv2.rectangle(dog, c1, c2, color, -1, lineType=cv2.LINE_AA) # -1 fill the rectangle
    cv2.putText(dog, label, (c1[0], c1[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)
    cv2.imshow("rectangle+text", dog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
