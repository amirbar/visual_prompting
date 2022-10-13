import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import os.path
from tqdm import trange
from evaluate.reasoning_dataloader import *
import cv2
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from tta import TTA, reverse_trans


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')
    parser.add_argument('--dataset_type', default='color')
    return parser


def get_default_mask_2rows_mask():
    mask = np.zeros((14,14))
    mask[:9] = 1
    mask[:, :7] = 1
    mask[: ,12:] = 1
    return mask


def _generate_result_for_canvas(args, model, inpt_pairs):
    """canvas is already in the right range."""

    final_imgs = []
    rcs_ls = [
        [TTA(shuffle_rows=False, shuffle_cols=False, transpose=False)],
        [TTA(shuffle_rows=False, shuffle_cols=True, transpose=True)],
        [TTA(shuffle_rows=False, shuffle_cols=True, transpose=True), TTA(shuffle_rows=False, shuffle_cols=False, transpose=False)],
        [TTA(shuffle_rows=False, shuffle_cols=True, transpose=True), TTA(shuffle_rows=False, shuffle_cols=False, transpose=True), TTA(shuffle_rows=False, shuffle_cols=False, transpose=False)],
        [TTA(shuffle_rows=False, shuffle_cols=False, transpose=True)]
    ][args.tta_option]
    for i in range(len(rcs_ls)):
        rcs = rcs_ls[i]
        canvas, len_keep_ps, ids_shuffle_ps, psuedo_gt_mask, v_order, shuffle_cols, transpose = rcs(
            inpt_pairs)
        input_image = torch.tensor(canvas).unsqueeze(0).to(args.device)
        _, im_paste, _ = generate_image(input_image, model, ids_shuffle_ps.to(args.device), len_keep_ps, device=args.device)
        im_paste = reverse_trans(im_paste, v_order, shuffle_cols, transpose)
        final_imgs.append(im_paste)
    if len(final_imgs) > 1:
        im_paste = np.mean(final_imgs, axis=0)
    else:
        im_paste = final_imgs[0]
    rcs = TTA(shuffle_rows=False, shuffle_cols=False, transpose=False)
    canvas, _, _, _, _, _, _ = rcs(inpt_pairs)
    input_image = torch.tensor(canvas).unsqueeze(0).to(args.device)
    canvas = torch.einsum('chw->hwc', input_image[0])
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)


def is_square(mask):
    mask = np.uint8(mask)
    contours,_ = cv2.findContours(mask.copy(), 1, 1) # not copying here will throw an error
    if not contours:
        return None
    x,y,w,h = cv2.boundingRect(contours[0])
    radius = max(h,w) // 2
    center_x = x + w/2
    center_y = y + h/2
    circle_mask = np.zeros_like(mask)
    circle_mask = cv2.circle(circle_mask, (int(center_x),int(center_y)), radius, 1, -1)
    circle_mask = circle_mask > 0
    square_mask = np.zeros_like(mask)
    square_mask[int(center_y)-radius: int(center_y)+radius, int(center_x)-radius:int(center_x)+radius] = 1
    square_mask = square_mask > 0
    mask = mask > 0
    square_shape = np.sum(np.float32(square_mask & mask)) / np.sum(np.float32(square_mask | mask))
    circle_shape = np.sum(np.float32(circle_mask & mask)) / np.sum(np.float32(circle_mask | mask))
    return square_shape > circle_shape

def calculate_metric(args, target, ours):
    # Crop the right area: 
    target = target[-74:, 113: 113+74]
    ours = ours[-74:, 113: 113+74]
    # Calculate accuracy: 
    accuracy = np.sum(np.float32((target == ours).all(axis=2))) / (ours.shape[0] * ours.shape[1])
    colors = np.unique(np.reshape(target, (-1, 3)), axis=0)
    assert colors.shape[0] == 2, colors # white and the expected color
    other_color = colors[0] if np.all(colors[1] == np.array([255,255,255])) else colors[1]
    seg_orig = ((target - other_color[np.newaxis, np.newaxis,:]) == 0).all(axis=2)
    seg_our = ((ours - other_color[np.newaxis, np.newaxis,:]) == 0).all(axis=2)
    color_blind_seg_our = (ours - np.array([[[255,255,255]]]) != 0).any(axis=2)
    iou = np.sum(np.float32(seg_orig & seg_our)) / np.sum(np.float32(seg_orig | seg_our))
    color_blind_iou = np.sum(np.float32(seg_orig & color_blind_seg_our)) / np.sum(np.float32(seg_orig | color_blind_seg_our))
    shape_accuracy = is_square(color_blind_seg_our)
    d = {
        'iou': iou, 
        'color_blind_iou': color_blind_iou, 
        'accuracy': accuracy,
    }
    if shape_accuracy is not None:
        d['shape_accuracy'] = shape_accuracy
    return d
    
def evaluate(args):
    with open(os.path.join(args.output_dir, 'log.txt'), 'w') as log:
        log.write(str(args)+'\n')

    ds = {
        'size': SizeChangeTask,
        'size_color': ChangeSizeColorTask,
        'size_shape': ChangeSizeShapeTask,
        'color': ColorChangeTask,
        'shape': ShapeChangeTask,
        'shape_color': ChangeShapeColorTask,
    }[args.dataset_type]()
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)
    # Build the transforms:
    figure_size = 74
    transforms = T.Compose([
        T.Resize((figure_size, figure_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    num_support = 2
    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0, 'shape_accuracy': 0}
    for idx in trange(len(ds)):
        query_image, query_target = ds[idx]
        pairs = []

        for k in range(num_support):

            idx2 = np.random.choice(np.arange(len(ds)))
            support_image, support_target = ds[idx2]
            pairs.append((support_image, support_target))

        pairs.append((query_image, query_target))

        inpt_pairs = []
        for p in pairs:
            support_image, support_target = p
            support_image_ten = transforms(Image.fromarray(support_image))
            support_target_ten = transforms(Image.fromarray(support_target))
            inpt_pairs.append((support_image_ten, support_target_ten))

        # Calculate the original_image and the result
        original_image, generated_result = _generate_result_for_canvas(args, model, inpt_pairs)
        original_image = round_image(original_image, ds.color_options() + [BLACK])
        generated_result = round_image(generated_result, ds.color_options()+ [BLACK])
        if args.output_dir:
            Image.fromarray(np.uint8(original_image)).save(
                os.path.join(args.output_dir, f'original_{idx}.png'))
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(args.output_dir, f'generated_{idx}.png'))
        current_metric = calculate_metric(args, original_image, generated_result)
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
            log.write(str(idx)+'\t'+str(current_metric)+'\n')
        for i, j in current_metric.items():
            eval_dict[i] += (j / len(ds))
        
    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
        log.write('all\t'+str(eval_dict)+'\n')
    

if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
