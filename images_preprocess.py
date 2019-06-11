import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image
from cache import cache
import json

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding

from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from helpers import load_json
from helpers import load_image
from helpers import print_progress

# only load the desired CNN model
from tensorflow.python.keras.applications import VGG16
#show_image(idx=1,train=True)
image_model = VGG16(include_top=True, weights='imagenet')
# we use the output of the final fc2 layer (fully connected dense layer)
transfer_layer=image_model.get_layer('fc2')
img_shape=(224,224)

#from tensorflow.python.keras.applications import InceptionV3
#image_model = InceptionV3(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('avg_pool')
#img_shape=(299,299)

#from tensorflow.python.keras.applications import ResNet50
#image_model = ResNet50(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('avg_pool')
#img_shape=(224,224)

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
img_size=K.int_shape(image_model.input)[1:3]

transfer_values_size=K.int_shape(transfer_layer.output)[1]



# function to show image along with captions
image_dir='../../../../Desktop/UAV/images/'
def show_image(idx, train):
    """
    Load and plot an image from the training- or validation-set
    with the given index.
    """

    if train:
        # Use an image from the training-set.
        dir = image_dir
        filename = filenames_train[idx]
        captions = captions_train[idx]
    else:
        # Use an image from the validation-set.
        dir = image_dir
        filename = filenames_val[idx]
        captions = captions_val[idx]

    # Path for the image-file.
    path = os.path.join(dir, filename)

    # Print the captions for this image.
    for caption in captions:
        print(caption)
    
    # Load the image and plot it.
    img = load_image(path)
    plt.imshow(img)
    plt.show()
    
# To process the images with VGG16 and getting the transfer values
def process_images(data_dir, filenames, batch_size=32):
    """
    Process all the given files in the given data_dir using the
    pre-trained image-model and return their transfer-values.
    
    Note that we process the images in batches to save
    memory and improve efficiency on the GPU.
    """
    
    # Number of images to process.
    num_images = len(filenames)

    # Pre-allocate input-batch-array for images.
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    # Initialize index into the filenames.
    start_index = 0

    # Process batches of image-files.
    while start_index < num_images:
        # Print the percentage-progress.
        print_progress(count=start_index, max_count=num_images)

        # End-index for this batch.
        end_index = start_index + batch_size

        # Ensure end-index is within bounds.
        if end_index > num_images:
            end_index = num_images

        # The last batch may have a different batch-size.
        current_batch_size = end_index - start_index

        # Load all the images in the batch.
        for i, filename in enumerate(filenames[start_index:end_index]):
            # Path for the image-file.
            path = os.path.join(data_dir, filename)

            # Load and resize the image.
            # This returns the image as a numpy-array.
            img = load_image(path,img_shape)

            # Save the image for later use.
            image_batch[i] = img

        # Use the pre-trained image-model to process the image.
        # Note that the last batch may have a different size,
        # so we only use the relevant images.
        transfer_values_batch = \
            image_model_transfer.predict(image_batch[0:current_batch_size])

        # Save the transfer-values in the pre-allocated array.
        transfer_values[start_index:end_index] = \
            transfer_values_batch[0:current_batch_size]

        # Increase the index for the next loop-iteration.
        start_index = end_index

    # Print newline.
    print()

    return transfer_values

# Process all images in training set
def process_images_train():
    print("Processing {0} images in training-set ...".format(len(filenames_train)))

    # Path for the cache-file.
    cache_path = os.path.join(image_dir,
                              "transfer_values_train.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=image_dir,
                            filenames=filenames_train)

    return transfer_values

# Process all images in evaluation set
def process_images_val():
    print("Processing {0} images in validation-set ...".format(len(filenames_val)))

    # Path for the cache-file.
    cache_path = os.path.join(image_dir,
                              "transfer_values_val.pkl")
    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=image_dir,
                            filenames=filenames_val)

    return transfer_values

def process_images_test():
    print("Processing {0} images in test-set ...".format(len(filenames_test)))

    # Path for the cache-file.
    cache_path = os.path.join(image_dir,
                              "transfer_values_test.pkl")
    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=image_dir,
                            filenames=filenames_test)

    return transfer_values

filenames_train=load_json('filenames_train_saifullah')
captions_train=load_json('captions_train_saifullah')

num_images_train=len(filenames_train)

filenames_val=load_json('filenames_val_saifullah')
captions_val=load_json('captions_val_saifullah')

filenames_test=load_json('filenames_test_saifullah')
captions_test=load_json('captions_test_saifullah')


# save files to pickle in data directory
def process_and_save():
    transfer_values_train=process_images_train()
    print("dtype:",transfer_values_train.dtype)
    print("shape:",transfer_values_train.shape)

    transfer_values_val=process_images_val()
    print("dtype:",transfer_values_val.dtype)
    print("shape:",transfer_values_val.shape)
