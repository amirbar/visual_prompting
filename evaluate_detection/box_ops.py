import torch
import cv2
import torch
import numpy as np


def to_rectangle(img, start_h=113, start_w=113):
    '''
    assuming image is a binary mask
    '''
    # from matplotlib import pyplot as plt

    img_np = img.numpy().astype('uint8')[start_h:, start_w:, 0]
    num_labels, labels_im = cv2.connectedComponents(img_np)
    new_img = np.zeros((img_np.shape[0], img_np.shape[1]))
    indices = np.argsort([np.sum(labels_im == i) for i in range(num_labels)])[::-1]
    for i in indices:
        indices_y, indices_x = np.where(labels_im == i)
        if img_np[indices_y[0], indices_x[0]] != 255:
            continue
        new_img[np.min(indices_y): np.max(indices_y) + 1, np.min(indices_x): np.max(indices_x) + 1] = 255
        break
    new_img = torch.tensor(new_img)
    new_img = torch.stack([new_img, new_img, new_img], dim=-1)
    img[start_h:, start_w:] = new_img

    return img

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

