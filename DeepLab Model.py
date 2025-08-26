import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from optuna.pruners import MedianPruner

import os
import glob
import random

IMAGE_SIZE = (256, 256)  #256 256

train_files = glob.glob(r'C:\Users\Jimmy\Downloads\slices\img/*.png')
mask_files = glob.glob(r'C:\Users\Jimmy\Downloads\slices\mask/*.png')

EPOCHS = 5
BATCH_SIZE = 8

def diagnosis(mask):
    value = np.max(cv2.imread(mask))
    return '1' if value > 0 else '0'

df = pd.DataFrame({"image_path": train_files,
                    "mask_path": mask_files,
                    "diagnosis": [diagnosis(x) for x in mask_files]})
df.head()

df_train, df_test = train_test_split(df, test_size=0.4)
df_test, df_val = train_test_split(df_train, test_size=0.5)
print(df_train.values.shape)
print(df_val.values.shape)
print(df_test.values.shape)

def create_generator(df, aug_dict, image_size = (256, 256), batch_size = 32, seed = 42):   #256 256

    IMG_SIZE = image_size
    BATCH_SIZE = batch_size
    SEED = seed

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_dataframe(
            df,
            class_mode = None,
            x_col = 'image_path',
            color_mode = 'rgb',
            target_size = IMG_SIZE,
            batch_size = BATCH_SIZE,
            save_prefix='image',
            seed = SEED)

    mask_generator = mask_datagen.flow_from_dataframe(
            df,
            class_mode = None,
            x_col = 'mask_path',
            color_mode = 'grayscale',
            target_size = IMG_SIZE,
            batch_size = BATCH_SIZE,
            save_prefix='mask',
            seed = SEED)

    generator = zip(image_generator, mask_generator)

    for (img,mask) in generator:
        img = img / 255
        msk = mask / 255
        msk[msk > 0.5] = 1
        msk[msk <= 0.5] = 0

        yield (img, msk)

train_aug_dict = dict(
    fill_mode='nearest')

train_generator = create_generator(df_train, aug_dict = train_aug_dict)
validation_generator = create_generator(df_val, aug_dict = {})
test_generator = create_generator(df_test, aug_dict = {})

images, masks = next(train_generator)
print(f"Number of images in the batch: {images.shape[0]}")
print(f"Shape of an image: {images.shape[1:]}")
print(f"Shape of a mask: {masks.shape[1:]}")

import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

smooth=1.

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)

IMAGE_SIZE = 256  #256
BATCH_SIZE = 8
NUM_CLASSES = 1

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False, l2_reg = 0.001):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(block_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model_input, outputs=model_output)

    return model

model = DeeplabV3(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()

opt = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
model.compile(optimizer=opt, loss=bce_dice_loss, metrics=["accuracy", iou, dice_coef])

callbacks = [
    ModelCheckpoint('unet.keras', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]

import math
steps_per_epoch = math.ceil(len(df_train) / BATCH_SIZE)
validation_steps = math.ceil(len(df_val) / BATCH_SIZE)

history_unet = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

model.save('TumorSeg_DeepLab_model.h5')
