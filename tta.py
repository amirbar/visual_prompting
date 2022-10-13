from evaluate.reasoning_dataloader import background_transforms
from evaluate.mae_utils import *
import torch
from PIL import Image
import numpy as np

DEVICE = 'cuda'
h, w = 224, 224


class RowColShuffle(torch.nn.Module):
    def __init__(self, shuffle_rows=False, shuffle_cols=False, num_rows=3):
        super(RowColShuffle, self).__init__()
        self.shuffle_rows = shuffle_rows
        self.shuffle_cols = shuffle_cols
        self.num_rows = num_rows

    def forward(self, pairs):
        background_image = Image.new('RGB', (224, 224), color='black')
        canvas = background_transforms(background_image)

        v_order = np.arange(0, self.num_rows)
        if self.shuffle_rows:
            np.random.shuffle(v_order)

        shuffle_cols = False
        if self.shuffle_cols:
            shuffle_cols = np.random.choice([True, False])

        padding = 1
        figure_size = 74
        for i in range(len(pairs)):
            img, label = pairs[v_order[i]]
            start_row = i * (figure_size + padding)
            if shuffle_cols:
                img, label = label, img
            canvas[:, start_row:start_row + figure_size, 224 // 2 - figure_size:224 // 2] = img
            canvas[:, start_row:start_row + figure_size, 224 // 2 + 1: 224 // 2 + 1 + figure_size] = label

        pred_row_idx = np.where(v_order == 2)[0][0]
        pred_col_idx = 1 if not shuffle_cols else 0
        canvas = np.array(canvas)

        # keep all but occluded part
        mask_psuedo_gt = np.ones((14, 14))
        row_mask_start = int(np.floor(14 * float(pred_row_idx) / 3))
        row_mask_end = int(np.ceil(14 * float(pred_row_idx + 1) / 3)) + 1
        mask_psuedo_gt[row_mask_start:row_mask_end, 2 + pred_col_idx * 5:2 + pred_col_idx * 5 + 5] = 0

        # keep everything in 20% except for the occluded part
        mask = np.round(np.random.uniform(0, 1, (14, 14)) >= 0.5)
        mask[row_mask_start:row_mask_end, 2 + pred_col_idx * 5:2 + pred_col_idx * 5 + 5] = 0

        _mask = obtain_values_from_mask(mask)
        _mask_psuedo_gt = obtain_values_from_mask(mask_psuedo_gt)

        return canvas, len(_mask), fill_to_full(_mask), mask, len(_mask_psuedo_gt), fill_to_full(
            _mask_psuedo_gt), mask_psuedo_gt

    def shuffle_cols(self, canvas, num_cols, h_order, fig_size, border_size):
        new_canvas = np.zeros_like(canvas)
        for i in range(num_cols):
            col_start = 224 - fig_size + i * (fig_size + border_size)
            original_col_start = 224 - fig_size + h_order[i] * (fig_size + border_size)
            new_canvas[:, :, col_start:col_start + fig_size] = canvas[:, :,
                                                               original_col_start: original_col_start + fig_size]
        return new_canvas

    def shuffle_rows(self, canvas, num_rows, v_order, fig_size, border_size):
        new_canvas = np.zeros_like(canvas)

        for i in range(num_rows):
            start_col = i * (fig_size + border_size)
            old_start_col = v_order[i] * (fig_size + border_size)
            new_canvas[:, start_col:start_col + fig_size] = canvas[:, old_start_col: old_start_col + fig_size]

        return new_canvas


def reverse_trans(im_paste, v_order, shuffle_cols, transpose):
    background_image = Image.new('RGB', (224, 224), color='black')
    new_canvas = np.array(background_image)

    if transpose:
        im_paste = np.transpose(im_paste, [1, 0, 2])

    padding = 1
    figure_size = 74
    for i in range(len(v_order)):
        start_row = i * (figure_size + padding)
        img = im_paste[start_row: start_row + figure_size, 224 // 2 - figure_size - 1: 224 // 2 - 1]
        label = im_paste[start_row:start_row + figure_size, 224 // 2 + 1: 224 // 2 + 1 + figure_size]

        if shuffle_cols:
            img, label = label, img

        start_row = v_order[i] * (figure_size + padding)
        new_canvas[start_row:start_row + figure_size, 224 // 2 - figure_size - 1:224 // 2 - 1] = img
        new_canvas[start_row:start_row + figure_size, 224 // 2 + 1: 224 // 2 + 1 + figure_size] = label
    return new_canvas


class TTA(torch.nn.Module):
    def __init__(self, shuffle_rows=False, shuffle_cols=False, transpose=False, num_rows=3):
        super(TTA, self).__init__()
        self.shuffle_rows = shuffle_rows
        self.shuffle_cols = shuffle_cols
        self.transpose = transpose
        self.num_rows = num_rows

    def forward(self, pairs):
        background_image = Image.new('RGB', (224, 224), color='black')
        canvas = background_transforms(background_image)

        v_order = np.arange(0, self.num_rows)
        if self.shuffle_rows:
            v_order = [2, 0, 1]

        shuffle_cols = False
        if self.shuffle_cols:
            shuffle_cols = True

        padding = 1
        figure_size = 74
        for i in range(len(pairs)):
            img, label = pairs[v_order[i]]
            start_row = i * (figure_size + padding)
            if shuffle_cols:
                img, label = label, img
            canvas[:, start_row:start_row + figure_size, 224 // 2 - figure_size - 1:224 // 2 - 1] = img
            canvas[:, start_row:start_row + figure_size, 224 // 2 + 1: 224 // 2 + 1 + figure_size] = label

        pred_row_idx = np.where(v_order == 2)[0][0]
        pred_col_idx = 1 if not shuffle_cols else 0
        canvas = np.array(canvas)

        # keep all but occluded part
        mask_psuedo_gt = np.ones((14, 14))
        row_mask_start = int(np.floor(14 * float(pred_row_idx) / 3))
        row_mask_end = int(np.ceil(14 * float(pred_row_idx + 1) / 3)) + 1
        mask_psuedo_gt[row_mask_start:row_mask_end, 2 + pred_col_idx * 5:2 + pred_col_idx * 5 + 5] = 0

        transpose_img = False
        if self.transpose:
            transpose_img = True

        if transpose_img:
            mask_psuedo_gt = np.transpose(mask_psuedo_gt, [1, 0])
            canvas = np.transpose(canvas, [0, 2, 1])

        _mask_psuedo_gt = obtain_values_from_mask(mask_psuedo_gt)
        return canvas, len(_mask_psuedo_gt), fill_to_full(
            _mask_psuedo_gt), mask_psuedo_gt, v_order, shuffle_cols, transpose_img

    def shuffle_cols(self, canvas, num_cols, h_order, fig_size, border_size):
        new_canvas = np.zeros_like(canvas)
        for i in range(num_cols):
            col_start = 224 - fig_size + i * (fig_size + border_size)
            original_col_start = 224 - fig_size + h_order[i] * (fig_size + border_size)
            new_canvas[:, :, col_start:col_start + fig_size] = canvas[:, :,
                                                               original_col_start: original_col_start + fig_size]
        return new_canvas

    def shuffle_rows(self, canvas, num_rows, v_order, fig_size, border_size):
        new_canvas = np.zeros_like(canvas)

        for i in range(num_rows):
            start_col = i * (fig_size + border_size)
            old_start_col = v_order[i] * (fig_size + border_size)
            new_canvas[:, start_col:start_col + fig_size] = canvas[:, old_start_col: old_start_col + fig_size]

        return new_canvas


