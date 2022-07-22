import cv2
import os, random
import numpy as np
from parameters import letters, max_text_len
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, Affine, RandomBrightnessContrast, CLAHE, ColorJitter, GaussianBlur
)
import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
# # Input data generator
def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: letters.index(x), text))


import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def elastic_transform(image, alpha_range, sigma, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)
        
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 channels, batch_size, downsample_factor, check_load_all, max_text_len=max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.channels = channels
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.check_load_all = check_load_all
        self.img_dirpath = img_dirpath                  # image dir path
        #self.img_dir = [f for f in os.listdir(img_dirpath) if f.endswith('.jpg') or f.endswith('.png')]     # images list
        self.img_dir = []
        for folder in os.listdir(self.img_dirpath):
          path = self.img_dirpath + folder
        #   print(path)
          for img in os.listdir(path): 
            img_path = path+'/'+img
            # print(img_path)
            if img.endswith('.jpg') or img.endswith('.png'):
                if os.stat(img_path).st_size == 0:
                    continue
                else: 
                    self.img_dir.append(img_path)
        
        self.n = len(self.img_dir)
        print(self.n)    
        # np.random.shuffle(self.img_dir)                  # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        if self.channels == 3 and self.check_load_all == True:
            self.imgs = np.zeros((self.n, self.img_h, self.img_w, self.channels))
        elif self.channels == 3 and self.check_load_all == False:
            self.imgs = None
        elif self.channels == 1 and self.check_load_all == True:
            self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        elif self.channels == 1 and self.check_load_all == False:
            self.imgs = None
        self.texts = []
    
    def aug_fn(self, image):
        transforms = Compose([
                            Rotate(limit=5),
                            # RandomBrightness(limit=0.05),
                            # JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
                            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                            # RandomContrast(limit=0.05, p=0.125),
                            ColorJitter (brightness=0.01, contrast=0.01, saturation=0.05, hue=0.05, always_apply=False, p=0.02),
                            RandomBrightnessContrast(p=0.05),
                            # GaussianBlur (blur_limit=(3, 7), sigma_limit=0.1, always_apply=False, p=0.1),

                            Affine(shear=(-5,5))
        ])
        aug_img = transforms(image=image)
        aug_img = aug_img['image']
        # aug_img = tf.cast(aug_img/255.0, tf.float32)
        # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
        return aug_img


    def build_data(self):
        if self.check_load_all: 

            print(self.n, " Image Loading start...")
            # print(len(self.img_dir))
            for i, img_file in enumerate(self.img_dir):
                try:
                    if self.channels == 1:
                        img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = cv2.resize(img, (self.img_w, self.img_h))
                    img = img.astype(np.float32)
                    img = (img / 255.0) * 2.0 - 1.0
                    
                    if self.channels == 3: 
                        self.imgs[i, :, :, :] = img
                    else:
                        self.imgs[i, :, :] = img
                    string = ''
                    img_file = img_file[0:len(img_file)-4]
                    for j in range(len(img_file)):
                        if img_file[j]>='0' and img_file[j]<='9':
                            continue
                        elif img_file[j]=='_':
                            string += ""
                        else:
                            string += img_file[j]
                    
                    # if len(image_file)!=11 and len(image_file)!=12:
                    #     print(image_file)
                    # if len(string) == 13:
                    #     pad = string 
                    #     string = 'Z' + pad    
                    # if len(string) == 12:
                    #     pad = string 
                    #     string = 'Z' + 'Z' + pad                        
                    # if len(string) == 11:
                    #     pad = string 
                    #     string = 'Z' + 'Z' + 'Z'+ pad                    
                    # if len(string) == 10:
                    #     pad = string 
                    #     string = 'Z' + 'Z' + 'Z' + 'Z'+ pad
                    if len(string) == 2:
                        pad = string 
                        string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ 'Z'+ pad
                    if len(string) == 3:
                        pad = string 
                        string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ pad
                    if len(string) == 4:
                        pad = string 
                        string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ pad
                    if len(string) == 5:
                        pad = string 
                        string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + pad
                    if len(string) == 6:
                        pad = string 
                        string = 'Z' + 'Z' + 'Z' + 'Z'+ pad
                    if len(string) == 7:
                        pad = string
                        string = 'Z' + 'Z' + 'Z'+ pad
                    if len(string) == 8:
                        pad = string
                        string = 'Z' + 'Z'+ pad
                    if len(string) == 9:
                        pad = string
                        string = 'Z'+ pad
                    self.texts.append(string)
                except:
                    print("fuck! " + str(img_file))
            print(len(self.texts))
            print(len(self.texts) == self.n)
            print(self.n, " Image Loading finish...")
        else:
            return None

    def next_sample(self):      ## index max -> 0 으로 만들기
        if self.check_load_all: 
            self.cur_index += 1

            if self.cur_index >= self.n:
                self.cur_index = 0
                random.shuffle(self.indexes)
            return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
        else:
            
            self.cur_index += 1
            if self.cur_index >= self.n:
                self.cur_index = 0
                random.shuffle(self.indexes)

            img_file_path = self.img_dir[self.indexes[self.cur_index]]


            if self.channels == 1:
                img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            string = ''
            img_file = img_file_path.split('/')[-1]
            img_file = img_file[0:len(img_file)-4]
            for j in range(len(img_file)):
                if img_file[j]>='0' and img_file[j]<='9':
                    continue
                elif img_file[j]=='_':
                    string += ""
                else:
                    string += img_file[j]
            # if len(image_file)!=11 and len(image_file)!=12:
            #     print(image_file)
            # if len(string) == 13:
            #     pad = string 
            #     string = 'Z' + pad    
            # if len(string) == 12:
            #     pad = string 
            #     string = 'Z' + 'Z' + pad                        
            # if len(string) == 11:
            #     pad = string 
            #     string = 'Z' + 'Z' + 'Z'+ pad                    
            # if len(string) == 10:
            #     pad = string 
            #     string = 'Z' + 'Z' + 'Z' + 'Z'+ pad
            # if len(string) == 9:
            #     pad = string 
            #     string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ pad
            # if len(string) == 8:
            #     pad = string 
            #     string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ pad
            # if len(string) == 7:
            #     pad = string 
            #     string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ 'Z' + pad
            # if len(string) == 6:
            #     pad = string 
            #     string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ 'Z' + 'Z' + pad
            # if len(string) == 5:
            #     pad = string
            #     string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ 'Z' + 'Z' + 'Z'+ pad
            # if len(string) == 4:
            #     pad = string
            #     string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ 'Z' + 'Z' + 'Z'+ 'Z'+ pad
            if len(string) == 2:
                pad = string 
                string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ 'Z'+ pad
            if len(string) == 3:
                pad = string 
                string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ pad
            if len(string) == 4:
                pad = string 
                string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + 'Z'+ pad
            if len(string) == 5:
                pad = string 
                string = 'Z' + 'Z' + 'Z' + 'Z' + 'Z' + pad
            if len(string) == 6:
                pad = string 
                string = 'Z' + 'Z' + 'Z' + 'Z'+ pad
            if len(string) == 7:
                pad = string
                string = 'Z' + 'Z' + 'Z'+ pad
            if len(string) == 8:
                pad = string
                string = 'Z' + 'Z'+ pad
            if len(string) == 9:
                pad = string
                string = 'Z'+ pad
            
            return img, string

    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, self.channels])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                

                choice = random.uniform(0,1)
                if choice > 0.3: 
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img = elastic_transform(img, alpha_range=90, sigma=6)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = self.aug_fn(img)

                # img = (img + 1.0)/2.0 *255.0
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.imwrite('/home/anlab/Desktop/fixing_3_channels/lpr/Pipeline/Model/Debug_batch/'+str(i)+'.jpg', img)
                if self.channels == 1: 
                    img = np.expand_dims(img, axis=-1)
                try:
                    img = np.transpose(img,(1,0,2))
                except Exception:
                    print(img.shape)

                img = np.expand_dims(img, axis=0)
                X_data[i] = img
                
                #if len(text_to_labels(text)) == 9:
                #    print(text)
                 
                if '_' in text:
                    print(text)
                    
                Y_data[i] = text_to_labels(text)


                label_length[i] = len(text)
            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length, 
                'label_length': label_length  
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
            # return inputs, outputs




    def next_batch_val(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, self.channels])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            

            for i in range(self.batch_size):
                img, text = self.next_sample()

                choice = random.uniform(0,1)
                if choice > 0.3: 
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img = elastic_transform(img, alpha_range=90, sigma=6)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if self.channels == 1: 
                    img = np.expand_dims(img, axis=-1)
                img = np.transpose(img,(1,0,2))
                img = np.expand_dims(img, axis=0)
                X_data[i] = img
                
                #if len(text_to_labels(text)) == 9:
                #    print(text)
                if '_' in text:
                    print(text)
                Y_data[i] = text_to_labels(text)
                label_length[i] = len(text)
                # X_data[i], Y_data[i] = self.process_data(X_data[i], Y_data[i])
            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length, 
                'label_length': label_length  
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)

# import ipdb
# ipdb.set_trace()
# train_file_path = '/home/anlab/Desktop/main_v3/lpr/Pipeline/train_200k/'
# tiger_train = TextImageGenerator(train_file_path, img_w, img_h, channels, batch_size, downsample_factor, check_load_all)
# tiger_train.build_data()
# list(tiger_train.next_batch())
