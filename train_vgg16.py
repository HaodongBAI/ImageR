from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras import backend as K
from PIL import Image
import numpy as np
import os
import h5py

K.set_image_dim_ordering('th')

def VGG_16(weights_path=None):

    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")) #(64, 112, 112)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")) #(128, 56, 56)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")) #(256, 28, 28)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")) #(512, 14, 14)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")) #(512, 7, 7)

    model.add(Flatten())
    model.add(Dense(4096, activation='relu')) #(4096)
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu')) #(4096)
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax')) #(1000)

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_model():
    model = VGG_16('vgg16_weights.h5')

    weights_path = 'vgg16_weights.h5'

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def load_image(url):
    class_dict = dict()
    image_db = dict()
    image_db['pixel'] = []
    image_db['label'] = []

    pathDir = os.listdir(url)

    for picname in pathDir[1:]:

        print picname + ' ',
        imgurl = url + picname
        img = Image.open(imgurl)
        im = np.array(img.resize((224, 224)), np.float32)
        print im.shape
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        # im = np.expand_dims(im, axis=0)
        print im.shape
        image_db['pixel'].append(im)

        label = filter(str.isalpha, picname)
        label = label[:-3]
        if label not in class_dict:
            class_dict[label] = len(class_dict)
        image_db['label'].append(class_dict[label])

    image_db['pixel'] = np.asarray(image_db['pixel'], dtype=np.int32)
    image_db['label'] = np.asarray(image_db['label'], dtype=np.int32)

    return image_db, class_dict

def predict(model, url):
    image_db, class_dict = load_image(url)
    out = model.predict(image_db['pixel'])
    print out.shape
    image_db['pred'] = out
    return image_db, class_dict
