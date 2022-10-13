import torch
import copy
import numpy as np
import matplotlib.patches as patches_plt
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def fill_to_full(arr):
    new_arr = copy.deepcopy(arr)
    if isinstance(new_arr, np.ndarray):
        new_arr = list(new_arr)
    for i in range(196):
        if i not in new_arr:
            new_arr.append(i)
    return torch.tensor(new_arr)[np.newaxis, ]


def fill_to_full_batched(arrs):
    new_arr = copy.deepcopy(arrs)
    if isinstance(new_arr, np.ndarray):
        new_arr = [list(n) for n in new_arr]
    for i in range(196):
        for k in new_arr:
            if i not in k:
                k.append(i)
    return torch.tensor(new_arr)


def convert_to_tensor(img):
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)
    if len(img.shape) != 4:
        # make it a batch-like
        img = img.unsqueeze(dim=0)
        img = torch.einsum('nhwc->nchw', img)
    elif img.shape[-1] == 3:
        assert isinstance(img, torch.Tensor)
        img = torch.einsum('nhwc->nchw', img)
    return img


def generate_mask_for_evaluation():
    mask = np.zeros((14,14))
    mask[:7] = 1
    mask[:, :7] = 1
    mask = obtain_values_from_mask(mask)
    len_keep = len(mask)
    return fill_to_full(mask), len_keep


def generate_mask_for_evaluation_2rows():
    mask = np.zeros((14,14))
    mask[:9] = 1
    mask[:, :7] = 1
    mask = obtain_values_from_mask(mask)
    len_keep = len(mask)
    return fill_to_full(mask), len_keep


def generate_mask_for_evaluation_2rows_more_context():
    mask = np.zeros((14,14))
    mask[:9] = 1
    mask[:, :7] = 1
    mask[: ,12:] = 1
    mask = obtain_values_from_mask(mask)
    len_keep = len(mask)
    return fill_to_full(mask), len_keep

    
def obtain_values_from_mask(mask: np.ndarray):
    if mask.shape == (14, 14):
        return list(mask.flatten().nonzero()[0])
    assert mask.shape == (224, 224)
    counter = 0
    values = []
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            if np.sum(mask[i:i+16, j:j+16]) == 16 ** 2:
                values.append(counter)
            counter += 1
    return values


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (0x44, 0x01, 0x54)
YELLOW = (0xFD, 0xE7, 0x25)


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', device='cpu'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.to(device)
    return model


@torch.no_grad()
def generate_image(orig_image, model, ids_shuffle, len_keep: int, device: str = 'cpu'):
    """ids_shuffle is [bs, 196]"""
    mask, orig_image, x = generate_raw_prediction(device, ids_shuffle, len_keep, model, orig_image)
    num_patches = 14
    y = x.argmax(dim=-1)
    im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)
    return orig_image, im_paste[0], mask


def decode_raw_predicion(mask, model, num_patches, orig_image, y):
    y = model.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
    y = model.vae.decode(y)
    # plt.figure(); plt.imshow(y[0].permute(1,2,0)); plt.show()
    y = F.interpolate(y, size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)
    y = torch.clip(y * 255, 0, 255).int().detach().cpu()
    # visualize the mask
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    orig_image = torch.einsum('nchw->nhwc', orig_image)
    orig_image = (
        torch.clip((orig_image[0].cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int()).unsqueeze(0)
    # MAE reconstruction pasted with visible patches
    im_paste = orig_image * (1 - mask) + y * mask
    return im_paste, mask, orig_image


@torch.no_grad()
def generate_raw_prediction(device, ids_shuffle, len_keep, model, orig_image):
    ids_shuffle = ids_shuffle.to(device)
    # make it a batch-like
    orig_image = convert_to_tensor(orig_image).to(device)
    temp_x = orig_image.clone().detach().to(device)
    # RUN ENCODER:
    # embed patches
    latent = model.patch_embed(temp_x.float())
    # add pos embed w/o cls token
    latent = latent + model.pos_embed[:, 1:, :]
    # masking: length -> length * mask_ratio
    N, L, D = latent.shape  # batch, length, dim
    # sort noise for each sample
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    latent = torch.gather(
        latent, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=latent.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # append cls token
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(latent.shape[0], -1, -1)
    latent = torch.cat((cls_tokens, latent), dim=1)
    # apply Transformer blocks
    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)
    x = model.decoder_embed(latent)
    # append mask tokens to sequence
    mask_tokens = model.mask_token.repeat(
        x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # add pos embed
    x = x + model.decoder_pos_embed
    # apply Transformer blocks
    for block_num, blk in enumerate(model.decoder_blocks):
        # Here is unrollment of the decoder blocks:
        x_temp = blk.norm1(x)
        # here is an unrollment of the attention mechanism:
        B, N, C = x_temp.shape
        qkv = blk.attn.qkv(x_temp).reshape(
            B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
        # The attention shape is [1, 16, 197, 197]
        # This is where our code comes to mind:
        attn = attn.softmax(dim=-1)

        x_temp = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_temp = blk.attn.proj(x_temp)
        x_temp = blk.attn.proj_drop(x_temp)
        # Here we continue to the orignal block.
        x = x + blk.drop_path(x_temp)

        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
    x = model.decoder_norm(x)
    # predictor projection
    x = model.decoder_pred(x)
    # remove cls token
    x = x[:, 1:, :]
    return mask, orig_image, x


@torch.no_grad()
def generate_decoder_embeddings(orig_image, model, ids_shuffle, len_keep, attribute: str = 'none', index: int = -1, device: str = 'cpu'):
    """ids_shuffle is [bs, 196]"""
    ids_shuffle = ids_shuffle.to(device)
    # make it a batch-like
    orig_image = convert_to_tensor(orig_image).to(device)
    temp_x = orig_image.clone().detach().to(device)

    # embed patches
    latent = model.patch_embed(temp_x.float())

    # add pos embed w/o cls token
    latent = latent + model.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    N, L, D = latent.shape  # batch, length, dim
    # sort noise for each sample
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    latent = torch.gather(
        latent, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=latent.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # append cls token
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(latent.shape[0], -1, -1)
    latent = torch.cat((cls_tokens, latent), dim=1)

    # apply Transformer blocks
    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)
    x = model.decoder_embed(latent)

    # append mask tokens to sequence
    mask_tokens = model.mask_token.repeat(
        x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # add pos embed
    x = x + model.decoder_pos_embed
    embeddings = []
    # apply Transformer blocks
    for block_num, blk in enumerate(model.decoder_blocks):
        # Here is unrollment of the decoder blocks:
        x_temp = blk.norm1(x)
        # here is an unrollment of the attention mechanism:
        B, N, C = x_temp.shape
        qkv = blk.attn.qkv(x_temp).reshape(
            B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        embeddings.append(
                (q.detach().cpu().numpy(), k.detach().cpu().numpy(), v.detach().cpu().numpy()))
        if block_num == index:
            return {
                'q': q.detach().cpu().numpy(), 
                'k': k.detach().cpu().numpy(), 
                'v': v.detach().cpu().numpy()
                }[attribute] 
        attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
        # The attention shape is [1, 16, 197, 197]
        # This is where our code comes to mind:
        attn = attn.softmax(dim=-1)

        x_temp = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_temp = blk.attn.proj(x_temp)
        x_temp = blk.attn.proj_drop(x_temp)
        # Here we continue to the orignal block.
        x = x + blk.drop_path(x_temp)

        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
    return embeddings


def show_image(image, title='', ax=None, patches=None, lines=None):
    # image is [H, W, 3]
    if ax is None:
        _, ax = plt.subplots()
    if patches is not None:
        for patch in patches:
            patch = [16 * (patch // 14), 16 *(patch % 14)]
            query = patches_plt.Rectangle(
                patch[::-1], 15, 15, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(query)
    assert image.shape[2] == 3
    if lines is not None:
        for line in lines:
            x, y = line
            x = [16 * (x // 14), 16 *(x % 14)]
            y = [16 * (y // 14), 16 *(y % 14)]
            plt.plot([y[1] + 8, x[1] + 8], [y[0] + 8, x[0] + 8], color="red", linewidth=1)
    ax.imshow(torch.clip((image.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def generate_for_training(orig_image, model, ids_shuffle, len_keep: int, device: str = 'cpu'):
    """ids_shuffle is [bs, 196]"""
    ids_shuffle = ids_shuffle.to(device)
    # make it a batch-like
    temp_x = orig_image

    # RUN ENCODER:
    # embed patches
    latent = model.patch_embed(temp_x.float())

    # add pos embed w/o cls token
    latent = latent + model.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    N, L, D = latent.shape  # batch, length, dim
    # sort noise for each sample
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    latent = torch.gather(
        latent, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # append cls token
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(latent.shape[0], -1, -1)
    latent = torch.cat((cls_tokens, latent), dim=1)

    # apply Transformer blocks
    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)
    x = model.decoder_embed(latent)

    # append mask tokens to sequence
    mask_tokens = model.mask_token.repeat(
        x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # add pos embed
    x = x + model.decoder_pos_embed

    # apply Transformer blocks
    for blk in model.decoder_blocks:
        x = blk(x)
    x = model.decoder_norm(x)

    # predictor projection
    x = model.decoder_pred(x)

    # remove cls token
    x = x[:, 1:, :]
    return x


@torch.no_grad()
def generate_decoder_attention_maps(orig_image, model, ids_shuffle, len_keep, index: int = -1, device: str = 'cpu'):
    """ids_shuffle is [bs, 196]"""
    ids_shuffle = ids_shuffle.to(device)
    # make it a batch-like
    orig_image = convert_to_tensor(orig_image).to(device)
    temp_x = orig_image.clone().detach().to(device)

    # embed patches
    latent = model.patch_embed(temp_x.float())

    # add pos embed w/o cls token
    latent = latent + model.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    N, L, D = latent.shape  # batch, length, dim
    # sort noise for each sample
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    latent = torch.gather(
        latent, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=latent.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # append cls token
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(latent.shape[0], -1, -1)
    latent = torch.cat((cls_tokens, latent), dim=1)

    # apply Transformer blocks
    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)
    x = model.decoder_embed(latent)

    # append mask tokens to sequence
    mask_tokens = model.mask_token.repeat(
        x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # add pos embed
    x = x + model.decoder_pos_embed
    embeddings = []
    attns = []
    # apply Transformer blocks
    for block_num, blk in enumerate(model.decoder_blocks):
        # Here is unrollment of the decoder blocks:
        x_temp = blk.norm1(x)
        # here is an unrollment of the attention mechanism:
        B, N, C = x_temp.shape
        qkv = blk.attn.qkv(x_temp).reshape(
            B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
        # The attention shape is [1, 16, 197, 197]
        # This is where our code comes to mind:
        attn = attn.softmax(dim=-1)
        attns.append(attn.detach().cpu().numpy())
        if block_num == index:
            return attns[-1]
        
        x_temp = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_temp = blk.attn.proj(x_temp)
        x_temp = blk.attn.proj_drop(x_temp)
        # Here we continue to the orignal block.
        x = x + blk.drop_path(x_temp)

        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
    return attns