from __future__ import division, print_function

import random

import PIL.Image
import torch
import torchvision
from torchvision import transforms


def std_per_channel(images):
  images = torch.stack(images, dim=0)
  return images.view(3, -1).std(dim=1)


def mean_per_channel(images):
  images = torch.stack(images, dim=0)
  return images.view(3, -1).mean(dim=1)


class Identity():  # used for skipping transforms
  def __call__(self, im):
    return im


class print_shape():
  def __call__(self, im):
    print(im.size)
    return im


class RGBToBGR():
  def __call__(self, im):
    assert im.mode == 'RGB'
    r, g, b = [im.getchannel(i) for i in range(3)]
    # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
    im = PIL.Image.merge('RGB', [b, g, r])
    return im


class pad_shorter():
  def __call__(self, im):
    h, w = im.size[-2:]
    s = max(h, w)
    new_im = PIL.Image.new("RGB", (s, s))
    new_im.paste(im, ((s-h)//2, (s-w)//2))
    return new_im


class ScaleIntensities():
  def __init__(self, in_range, out_range):
    """ Scales intensities. For example [-1, 1] -> [0, 255]."""
    self.in_range = in_range
    self.out_range = out_range

  def __oldcall__(self, tensor):
    tensor.mul_(255)
    return tensor

  def __call__(self, tensor):
    tensor = (
        tensor - self.in_range[0]
    ) / (
        self.in_range[1] - self.in_range[0]
    ) * (
        self.out_range[1] - self.out_range[0]
    ) + self.out_range[0]
    return tensor


def make_transform(is_train=True, is_inception=False):
  resnet_resize = 256
  resnet_cropsize = 224
  resnet_mean = [0.485, 0.456, 0.406]
  resnet_std = [0.229, 0.224, 0.225]

  if is_train:
    resnet_transform = transforms.Compose([
        transforms.RandomResizedCrop(resnet_cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])
  else:
    resnet_transform = transforms.Compose([
        transforms.Resize(resnet_resize),
        transforms.CenterCrop(resnet_cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

  return resnet_transform
