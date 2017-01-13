# coding=utf-8
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
import keras.backend as K
import numpy as np
import os
import timeit
import cv2
import h5py
import cPickle as pickle


def load_caltech(cls, url='static/caltech101'):
    class_dict = dict()
    image_db = dict()
    image_db['pixel'] = []
    image_db['label'] = []

    assert os.path.exists(url), 'Original images not found (see "url" variable in script).'

    pathDir = os.listdir(url)
    for clsname in [pathDir[cls]]:
        clsPath = os.listdir(url + '/' + clsname)

        label = len(class_dict)
        class_dict[label] = clsname
        print 'loading class ' + clsname[:] + ' ...'

        for picname in clsPath[:]:
            imgurl = url + '/' + clsname + '/' + picname
            img = cv2.imread(imgurl)
            im = np.array(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), np.float32)
            im[:, :, 0] -= 103.939
            im[:, :, 1] -= 116.779
            im[:, :, 2] -= 123.68
            im = im.transpose((2, 0, 1))
            image_db['pixel'].append(im)
            image_db['label'].append(label)

    image_db['pixel'] = np.asarray(image_db['pixel'])
    image_db['label'] = np.asarray(image_db['label'])

    return image_db, class_dict


def vgg_bottom(weights_path='static/params/vgg16_weights.h5'):
    K.set_image_dim_ordering('th')

    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))  # (64, 112, 112)

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))  # (128, 56, 56)

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))  # (256, 28, 28)

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))  # (512, 14, 14)

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))  # (512, 7, 7)

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))  # (4096)
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))  # (4096)
    model.add(Dropout(0.5))

    # print len(model.layers)

    # loading the weights of the pre-trained VGG16:
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded...')

    return model


def preprocess_db():
    model = vgg_bottom()

    for c in range(1, 102):
        print 'Processing {} cls...'.format(c)
        image_db, class_dict = load_caltech(c=c)
        print '\t{} images in this cls'.format(len(image_db['pixel']))
        start_time = timeit.default_timer()
        out = model.predict(image_db['pixel'])
        end_time = timeit.default_timer()

        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)))

        print '\tOutput shape: ' + str(out.shape)
        image_db['pred'] = out

        print 'start to pack pred...'
        fp = open('static/preprocess/pred_{}.pkl'.format(c), 'wb')
        pickle.dump(image_db['pred'], fp, -1)
        fl = open('static/preprocess/label_{}.pkl'.format(c), 'wb')
        pickle.dump(image_db['label'], fl, -1)
        print '\tfinish packing pred...\n'

