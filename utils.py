"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import cv2
import os

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=128):
    img = imread(image_path)
    # img = cv2.resize(img, (600, 300))
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    # mask_A = np.load('datasets/personReid/trainA_mask/'+image_path[0].split('/')[-1].split('.')[0]+'.npy')
    # mask_B = np.load('datasets/personReid/trainB_mask/'+image_path[1].split('/')[-1].split('.')[0]+'.npy')

    duke_root = r"F:\data_set\zipai\changji1_"
    market_root = r"F:\data_set\zipai\changjin2_"
    img_duke = os.path.split(image_path[0])
    img_market = os.path.split(image_path[1])

    # img_A = cv2.resize(img_A, (128, 128))
    # img_B = cv2.resize(img_B, (128, 128))
    # print(img_duke[1])
    # print(img_market[1])
    seg_duke = cv2.imread(os.path.join(duke_root, img_duke[1].split(".")[0]+".jpg"))
    seg_market = cv2.imread(os.path.join(market_root, img_market[1].split(".")[0]+".jpg"))
    if not is_testing:

        # seg_duke = cv2.resize(seg_duke, (128, 128))
        # seg_market = cv2.resize(seg_market, (128, 128))
        # print(seg_duke[:, :, 0].shape)
        # print(seg_market[:, :, 0].shape)

        # img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        # img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        #
        #
        # h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        # w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        # img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        # img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            seg_duke = np.fliplr(seg_duke)
            seg_market = np.fliplr(seg_market)
    # else:
        # img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        # img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])


    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    
    img_AB = np.concatenate((img_A, img_B), axis=2)
    # mask_A = mask_A.reshape(256,256,1)
    # mask_B = mask_B.reshape(256,256,1)
    # if not is_testing:
    img_AB = np.concatenate((img_AB, seg_duke[:, :, 0].reshape(600, 300, 1)), axis=2)
    img_AB = np.concatenate((img_AB, seg_market[:, :, 0].reshape(600, 300, 1)), axis=2)

    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim+2)
    return img_AB


def load_test_data1(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    # mask_A = np.load('datasets/personReid/trainA_mask/'+image_path[0].split('/')[-1].split('.')[0]+'.npy')
    # mask_B = np.load('datasets/personReid/trainB_mask/'+image_path[1].split('/')[-1].split('.')[0]+'.npy')

    duke_root = r"F:\data_set\DukeMTMC-reID\DukeMTMC-reID\c1_to_\seg_crf01"
    market_root = r"E:\dataset_for_ubantu\datasets\Market-1501-v15.09.15\c1_to_c6\seg_crf"
    img_duke = os.path.split(image_path[0])
    img_market = os.path.split(image_path[1])

    # img_A = cv2.resize(img_A, (128, 128))
    # img_B = cv2.resize(img_B, (128, 128))
    # print(img_duke[1])
    # print(img_market[1])
    seg_duke = cv2.imread(os.path.join(duke_root, img_duke[1]))
    seg_market = cv2.imread(os.path.join(market_root, img_market[1]))
    if not is_testing:

        # seg_duke = cv2.resize(seg_duke, (128, 128))
        # seg_market = cv2.resize(seg_market, (128, 128))
        # print(seg_duke[:, :, 0].shape)
        # print(seg_market[:, :, 0].shape)

        # img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        # img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        #
        #
        # h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        # w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        # img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        # img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            seg_duke = np.fliplr(seg_duke)
            seg_market = np.fliplr(seg_market)
            # else:
            # img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
            # img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A / 127.5 - 1.
    img_B = img_B / 127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # mask_A = mask_A.reshape(256,256,1)
    # mask_B = mask_B.reshape(256,256,1)
    # if not is_testing:
    img_AB = np.concatenate((img_AB, seg_duke[:, :, 0].reshape(600, 300, 1)), axis=2)
    img_AB = np.concatenate((img_AB, seg_market[:, :, 0].reshape(600, 300, 1)), axis=2)

    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim+2)
    return img_AB, img_duke, img_market
# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
