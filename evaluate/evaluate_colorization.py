import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import os.path

import torchvision
from tqdm import trange
from evaluate.in_colorization_dataloader import DatasetColorization
from evaluate.reasoning_dataloader import *
from evaluate.mae_utils import *
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--data_path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')

    parser.set_defaults(autoregressive=False)
    return parser


def _generate_result_for_canvas(args, model, canvas):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _ = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device)
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)


def calculate_metric(args, target, ours):
    ours = (np.transpose(ours/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    target = (np.transpose(target/255., [2, 0, 1]) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = np.mean((target - ours)**2)
    return {'mse': mse}


def evaluate(args):
    with open(os.path.join(args.output_dir, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')

    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)
    # Build the transforms:
    padding = 1

    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop((224 // 2 - padding, 224 // 2 - padding)),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop((224 // 2 - padding, 224 // 2 - padding)),
         torchvision.transforms.Grayscale(3),
         torchvision.transforms.ToTensor()])

    ds = DatasetColorization(args.data_path, image_transform, mask_transform)

    eval_dict = {'mse': 0.}

    for idx in trange(len(ds)):
        canvas = ds[idx]['grid']
        canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        original_image, generated_result = _generate_result_for_canvas(args, model, canvas)

        if args.output_dir:
            Image.fromarray(np.uint8(original_image)).save(
                os.path.join(args.output_dir, f'original_{idx}.png'))
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(args.output_dir, f'generated_{idx}.png'))

        if args.output_dir:
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(args.output_dir, f'generated_before_rounding_{idx}.png'))
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(args.output_dir, f'generated_rounded_{idx}.png'))
            Image.fromarray(np.uint8(original_image)).save(
                os.path.join(args.output_dir, f'original_{idx}.png'))
            Image.fromarray(np.uint8(generated_result)).save(
                os.path.join(args.output_dir, f'generated_fixed_{idx}.png'))

        current_metric = calculate_metric(args, original_image, generated_result)
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
            log.write(str(idx) + '\t' + str(current_metric) + '\n')
        for i, j in current_metric.items():
            eval_dict[i] += (j / len(ds))

    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
        log.write('all\t' + str(eval_dict) + '\n')


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
