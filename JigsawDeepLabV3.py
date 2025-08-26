import pandas as pd
import numpy as np
import glob
import math
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
import cv2
import itertools

# SETTINGS
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS_SEG = 5
EPOCHS_JIGSAW = 3
GRID_SIZE = 3

train_files = glob.glob(r'C:\Users\Jimmy\Downloads\slices\img/*.png')
mask_files = glob.glob(r'C:\Users\Jimmy\Downloads\slices\mask/*.png')

# DATAFRAME
def diagnosis(mask):
    value = np.max(cv2.imread(mask))
    return '1' if value > 0 else '0'

df = pd.DataFrame({"image_path": train_files,
                   "mask_path": mask_files,
                   "diagnosis": [diagnosis(x) for x in mask_files]})

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.4)
df_test, df_val = train_test_split(df_train, test_size=0.5)

# IMAGE GENERATOR
def create_generator(df, batch_size = 32, image_size = (IMAGE_SIZE, IMAGE_SIZE)):
    datagen = ImageDataGenerator()
    img_gen = datagen.flow_from_dataframe(df, x_col='image_path', class_mode=None,
                                          target_size=image_size, batch_size=batch_size, seed=42)
    mask_gen = datagen.flow_from_dataframe(df, x_col='mask_path', class_mode=None,
                                           color_mode='grayscale', target_size=image_size,
                                           batch_size=batch_size, seed=42)
    for img, mask in zip(img_gen, mask_gen):
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        yield img/255.0, mask/255.0

train_generator = create_generator(df_train, batch_size=BATCH_SIZE)
val_generator = create_generator(df_val, batch_size=BATCH_SIZE)

# METRICS & LOSS
smooth=1.
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    return (2.*intersection + smooth)/(K.sum(y_true)+K.sum(y_pred)+smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return dice_coef(y_true, y_pred) + bce

# DEEPLABV3
def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                      padding='same', use_bias=False, kernel_initializer='he_normal')(block_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = convolution_block(x, kernel_size=1)
    x = layers.UpSampling2D(size=(dims[1] // x.shape[1], dims[2] // x.shape[2]), interpolation='bilinear')(x)
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    x = layers.Concatenate(axis=-1)([x, out_1, out_6, out_12, out_18])
    x = convolution_block(x, kernel_size=1)
    return x

def DeeplabV3(backbone, image_size=IMAGE_SIZE):
    x = backbone.get_layer('conv4_block6_2_relu').output
    x = DilatedSpatialPyramidPooling(x)
    input_a = layers.UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]))(x)
    input_b = backbone.get_layer('conv2_block3_2_relu').output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]))(x)
    output = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    model = Model(inputs=backbone.input, outputs=output)
    return model

# JIGSAW PRETRAINING
def load_images(df):
    imgs = []
    for f in df['image_path']:
        img = tf.io.read_file(f)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        imgs.append(img/255.0)
    return np.stack(imgs)

def create_jigsaw_full(images, grid_size=GRID_SIZE):
    N, H, W, C = images.shape
    patch_h, patch_w = H//grid_size, W//grid_size
    X, Y = [], []
    perms = list(itertools.permutations(range(grid_size**2)))
    for img in images:
        patches = [img[i*patch_h:(i+1)*patch_h,j*patch_w:(j+1)*patch_w,:]
                   for i in range(grid_size) for j in range(grid_size)]
        perm = perms[np.random.randint(len(perms))]
        shuffled = np.concatenate([patches[k] for k in perm], axis=-1)
        X.append(shuffled)
        Y.append(perms.index(perm))
    return np.array(X), np.array(Y), len(perms)

all_images = load_images(df)
jigsaw_X, jigsaw_Y, NUM_CLASSES = create_jigsaw_full(all_images)

# Jigsaw model
jigsaw_input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3*GRID_SIZE**2)
inputs = layers.Input(shape=jigsaw_input_shape)
x = layers.Conv2D(64,3,activation='relu',padding='same')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(512,activation='relu')(x)
outputs = layers.Dense(NUM_CLASSES,activation='softmax')(x)
jigsaw_model = Model(inputs, outputs)
jigsaw_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
jigsaw_model.fit(jigsaw_X, jigsaw_Y, epochs=EPOCHS_JIGSAW, batch_size=BATCH_SIZE)

# TRANSFER TO BACKBONE
resnet_backbone = ResNet50(weights=None, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
for i in range(1, len(resnet_backbone.layers)):
    try:
        resnet_backbone.layers[i].set_weights(jigsaw_model.layers[min(i,len(jigsaw_model.layers)-1)].get_weights())
    except:
        pass

# FINAL DEEPLABV3 TRAINING
model = DeeplabV3(resnet_backbone, image_size=IMAGE_SIZE)
model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=['accuracy', dice_coef])
steps = math.ceil(len(df_train)/BATCH_SIZE)
val_steps = math.ceil(len(df_val)/BATCH_SIZE)
callbacks = [
    ModelCheckpoint('deeplab_jigsaw_full.h5', save_best_only=True, verbose=1),
    ReduceLROnPlateau(patience=5),
    EarlyStopping(patience=10)
]

history = model.fit(train_generator, steps_per_epoch=steps, epochs=EPOCHS_SEG,
                    validation_data=val_generator, validation_steps=val_steps,
                    callbacks=callbacks)
