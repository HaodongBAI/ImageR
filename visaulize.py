#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pylab
import os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
import load_data as op
import cPickle as pickle
import train_vgg16 as vgg16
import cv


# def changeArrayToImage():
#     dataset = op.unpickle('cifar/data_batch_1')
#     img = dataset['data'][0]
#     img = img.reshape((3, 32, 32))
#     img = np.transpose(img, (1, 2, 0))
#     img = Image.fromarray(img)
#     img.show()
#     # plot original image and first and second components of output
#     pylab.subplot(5, 5, 1); pylab.axis('off'); pylab.imshow(img)
#     pylab.gray();
#     # recall that the convOp output (filtered image) is actually a "minibatch",
#     # of size 1 here, so we take index 0 in the first dimension:
#     pylab.subplot(5, 5, 2); pylab.axis('off'); pylab.imshow(img)
#     pylab.subplot(5, 5, 3); pylab.axis('off'); pylab.imshow(img)
#     pylab.show()
#
#
# def array2Img(array, img_shape):
#     img = array.reshape(img_shape)
#     img = np.transpose(img, (1, 2, 0))
#     img = Image.fromarray(img)
#     img.show()
#     return img
#
#
# def readImage(filename, shape):
#     img = Image.open(filename).resize(shape)
#     img_array = np.array(img)
#     img_array = np.transpose(img_array, (2, 0, 1))
#     img_array = np.reshape(img_array, 3 * shape[0] * shape[1])
#     return img_array
#
#
# def clear_blackwrite():
#     url = '/Users/Kevin/Desktop/image/'
#     pathDir = os.listdir(url)
#     rm_list = []
#     for pic in pathDir[1:]:
#         img = Image.open(url + pic)
#         im = np.array(img.resize((224, 224)), np.float32)
#         if im.shape != (224, 224, 3):
#             rm_list.append(url + pic)
#
#     for file in rm_list:
#         os.remove(file)
#
#     print 'finish rm...'


def calc_pred_vector():
    url = r'/Users/Kevin/Desktop/image/'
    print 'start to load model...'
    model = vgg16.load_model()
    print 'finish loading model...'
    print 'start to pred pic...'
    image_db, class_dict = vgg16.predict(model, url)
    print 'finish preding pic...'

    print image_db['pred'].shape, image_db['label'].shape

    print 'start to pack pred...'
    fp = open('pred.pkl', 'wb')
    pickle.dump(image_db['pred'], fp, -1)
    fl = open('label.pkl', 'wb')
    pickle.dump(image_db['label'], fl, -1)
    print 'finish packing pred...'


def tsne_decom(image_db):
    print 'start tsne...'
    print image_db['pred'].shape

    X_tsne = TSNE(n_components=3, learning_rate=100).fit_transform(image_db['pred'])

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=image_db['label'])
    plt.show()



image_db = dict()
image_db['pred'] = pickle.load(open('pred.pkl'))
image_db['label'] = pickle.load(open('label.pkl'))

imgurl = r'/Users/Kevin/Desktop/image/apple10.jpg'
img = Image.open(imgurl)
im = np.array(img.resize((224, 224)), np.float32)
print im.shape
im[:, :, 0] -= 103.939
im[:, :, 1] -= 116.779
im[:, :, 2] -= 123.68
im = im.transpose((2, 0, 1))
im = np.expand_dims(im, axis=0)

model = vgg16.load_model()
out = model.predict(im)

def similarity(img, image_db):
    sim = np.dot(image_db['pred'], img.T)
    print sim
    rank = np.argsort(-sim,axis=0)
    print rank
    print image_db['pred']
    print sim[rank].T
    print image_db['label'][rank].T

similarity(out, image_db)