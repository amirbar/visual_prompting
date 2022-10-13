import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')
import torch
from PIL import Image
from torchvision.transforms import transforms, ToPILImage
from glob import glob
from matplotlib import pyplot as plt
import numpy as np

demo_images = glob("./imgs/*")
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

t = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


@torch.no_grad()
def get_demo_predictions(args, device, model):
    figs = get_demo_predictions_with_mask(args, model, t)
    return {"image_%s" % i: fig for i, fig in enumerate(figs)}


def show_image(image, ax, in_reverse=True):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(image, vmin=0, vmax=255)
    ax.axis('off')
    return


@torch.no_grad()
def get_demo_predictions_with_mask(args, model, t):
    num_patches = 14
    imgs = []
    for p in glob("./imgs/*"):
        with open(p, 'rb') as f:
            png = Image.open(f).convert('RGBA')
            background = Image.new('RGBA', png.size, (255, 255, 255))
            img = Image.alpha_composite(background, png).convert('RGB').resize((args.input_size, args.input_size),
                                                                               resample=Image.LANCZOS)
            img = t(img)
            imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    x = imgs.cuda(non_blocking=True)
    _, y, mask = model(x.float(), mask_ratio=0.75)
    y = y.argmax(dim=-1)
    y = model.vae.quantize.get_codebook_entry(y.reshape(-1), [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
    y = model.vae.decode(y).detach().cpu()
    y = F.interpolate(y, size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)
    y = torch.clip(y * 255, 0, 255).int()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x).to(mask)

    # masked image
    im_masked = x * (1 - mask)
    im_masked = torch.clip((im_masked * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    # MAE reconstruction pasted with visible patches
    x = torch.clip((x * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    im_paste = (x * (1 - mask) + y * mask).int()

    # make the plt figure larger
    # plt.figure()
    figs = []
    for k in range(0, len(imgs), 4):
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(len(imgs[k:k + 4])):
            show_image(x[k + i], ax[i, 0])
            show_image(im_masked[k+i], ax[i, 1])
            show_image(y[k + i], ax[i, 2], in_reverse=False)
            show_image(im_paste[k+i], ax[i, 3])

            for j in range(4):
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                ax[i, j].set_aspect('equal')
        figs.append(fig)

    # plt.show()

    return figs
