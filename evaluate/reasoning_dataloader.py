import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import numpy as np

h, w = 224, 224


def get_img(center_coordinates=None, radius=None, color=None, shape=None):
    image = Image.new('RGB', size=(224, 224), color='white')
    w, h = image.size
    if center_coordinates is None:
        center_coords_x = np.random.randint(10, w-10)
        center_coords_y = np.random.randint(10, h-10)
        center_coordinates = (center_coords_x, center_coords_y)

    if radius is None:
        radius = np.random.randint(10, 50)

    if color is None:
        color = (0, 255, 0)

    thickness = -1
    image = np.array(image)

    if shape is None or shape == 'circle':
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
    elif shape == 'rectangle':
        start_point = (center_coordinates[0] -
                       radius, center_coordinates[1] - radius)
        end_point = (center_coordinates[0] +
                     radius, center_coordinates[1] + radius)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    else:
        raise ValueError("Wrong shape")

    return image, center_coordinates, radius, color


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def round_image(img, options=(WHITE, BLACK, RED, GREEN, BLUE), outputs=None, t=(0, 0, 0)):
    # img.shape == [224, 224, 3], img.dtype == torch.int32
    img = torch.tensor(img)
    t = torch.tensor((t)).to(img)
    options = torch.tensor(options)
    opts = options.view(len(options), 1, 1, 3).permute(1, 2, 3, 0).to(img)
    nn = (((img + t).unsqueeze(-1) - opts) ** 2).float().mean(dim=2)
    nn_indices = torch.argmin(nn, dim=-1)
    if outputs is None:
        outputs = options
    res_img = torch.tensor(outputs)[nn_indices]
    return res_img

# fixed as circle
class ColorChangeTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ColorChangeTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [BLUE, GREEN, WHITE]

    def __getitem__(self, index):
        image1, center_coordinates1, radius1, color1 = get_img(
            color=GREEN)
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates1, radius1, BLUE)  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2

class SizeChangeTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(SizeChangeTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [BLUE, GREEN, WHITE]

    def __getitem__(self, index):
        radius1 = 30
        radius2 = 20
        image1, center_coordinates1, radius1, color1 = get_img(color=GREEN, shape='circle', radius=radius1)
        image2, center_coordinates2, radius2, color1 = get_img(
            center_coordinates1, radius2, GREEN, shape='circle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2

class ShapeChangeTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ShapeChangeTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, WHITE]

    def __getitem__(self, index):
        image1, center_coordinates1, radius1, color1 = get_img(
            color=GREEN, shape='circle')
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates1, radius1, color1, shape='rectangle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


# This task is actually very hard for MAEVQGAN or any other inpainting task. We discussed that issue in the limitations.
class ChangeLocationTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeLocationTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, WHITE]

    def __getitem__(self, index):
        image1, center_coordinates1, radius1, color1 = get_img(
            color=GREEN, shape='circle')
        center_coordinates2 = (
            223 - center_coordinates1[0], center_coordinates1[1])
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates2, radius1, color1, shape='circle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


class ChangeLocationVFlipTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeLocationVFlipTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, WHITE]

    def __getitem__(self, index):
        image1, center_coordinates1, radius1, color1 = get_img(
            color=GREEN, shape='circle')
        center_coordinates2 = (
            center_coordinates1[0], 223 - center_coordinates1[1])
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates2, radius1, color1, shape='circle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


class ChangeLocationTransposeTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeLocationTransposeTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, WHITE]

    def __getitem__(self, index):
        image1, center_coordinates1, radius1, color1 = get_img(
            color=GREEN, shape='circle')
        image1 = Image.fromarray(image1)
        image2 = np.array(image1.transpose(5))
        image1 = np.array(image1)
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


class ChangeLocationHShift(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeLocationHShift, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, WHITE]

    def __getitem__(self, index):
        h = w = 224
        shift = 50
        center_coords_x = np.random.randint(10, w - 10 - shift)
        center_coords_y = np.random.randint(10, h - 10)
        center_coordinates = (center_coords_x, center_coords_y)

        image1, center_coordinates1, radius1, color1 = get_img(
            center_coordinates, color=GREEN, shape='circle')
        center_coordinates2 = (
            center_coordinates[0] + shift, center_coordinates1[1])
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates2, radius1, color1, shape='circle')  # Get boxes

        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


class ChangeShapeColorTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeShapeColorTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, BLUE, WHITE]

    def __getitem__(self, index):
        image1, center_coordinates1, radius1, color1 = get_img(
            color=GREEN, shape='circle')
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates1, radius1, BLUE, shape='rectangle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


class ChangeLocationColorTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeLocationColorTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, BLUE, WHITE]

    def __getitem__(self, index):
        h = w = 224
        shift = 50
        center_coords_x = np.random.randint(10, w - 10 - shift)
        center_coords_y = np.random.randint(10, h - 10)
        center_coordinates = (center_coords_x, center_coords_y)

        image1, center_coordinates1, radius1, color1 = get_img(
            center_coordinates, color=GREEN, shape='circle')
        center_coordinates2 = (
            center_coordinates[0] + shift, center_coordinates1[1])
        image2, center_coordinates2, radius2, color2 = get_img(
            center_coordinates2, radius1, color=BLUE, shape='circle')  # Get boxes

        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2

class ChangeSizeColorTask(Dataset):

    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeSizeColorTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, BLUE, WHITE]

    def __getitem__(self, index):
        radius1 = 30
        radius2 = 20
        image1, center_coordinates1, radius1, color1 = get_img(color=GREEN, shape='circle', radius=radius1)
        image2, center_coordinates2, radius2, color1 = get_img(
            center_coordinates1, radius2, BLUE, shape='circle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


class ChangeSizeShapeTask(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        super(ChangeSizeShapeTask, self).__init__()

    def __len__(self):
        return 100

    def color_options(self, ):
        """The color options for the output. We take all of the colors that are in the image as possiblities"""
        return [GREEN, WHITE]

    def __getitem__(self, index):
        radius1 = 30
        radius2 = 20
        image1, center_coordinates1, radius1, color1 = get_img(color=GREEN, shape='circle', radius=radius1)
        image2, center_coordinates2, radius2, color1 = get_img(
            center_coordinates1, radius2, GREEN, shape='rectangle')  # Get boxes
        if self.transforms is not None:
            return self.transforms(Image.fromarray(image1)), self.transforms(Image.fromarray(image2))
        return image1, image2


def box_to_img(mask, target, border_width=4):
    if mask is None:
        mask = np.zeros((112, 112, 3))
    h, w, _ = mask.shape
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = list(
            (box * (h - 1)).round().int().numpy())
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max),
                      (255, 255, 255), border_width)
    return Image.fromarray(mask.astype('uint8'))


def get_annotated_image(img, boxes, border_width=3, mode='draw', copy_img=True):
    if mode == 'draw':
        h, w, _ = img.shape
        if copy_img:
            image_copy = np.array(img.copy())
        else:
            image_copy = np.array(Image.new('RGB', (w, h), color='black'))

        for box in boxes:
            box = box.numpy().astype('int')
            cv2.rectangle(
                image_copy, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), border_width)
    elif mode == 'keep':
        h, w, _ = img.shape
        image_copy = np.array(Image.new('RGB', (w, h), color='white'))

        for box in boxes:
            box = box.numpy().astype('int')
            image_copy[box[1]:box[3], box[0]:box[2]
                       ] = img[box[1]:box[3], box[0]:box[2]]

    return image_copy


background_transforms = T.Compose([
    T.Resize((224, 224)),
    T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
])


def create_grid_from_images(canvas, pairs, padding, figure_size):
    
    for i in range(len(pairs)):
        img, label = pairs[i]
        start_row = i*(figure_size + padding)
        
        canvas[:, start_row:start_row + figure_size, 224//2 - figure_size :224//2] = img
        canvas[:, start_row:start_row + figure_size, 224//2 +1 : 224//2 +1 + figure_size] = label

    return canvas

if __name__ == "__main__":
    ds = ChangeLocationTransposeTask()
    img1, img2 = ds[10]