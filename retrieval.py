# coding=utf-8
import numpy as np
import os
import cPickle as pickle
import cv2
import preprocesser as PP
import reducer as RD
import cluster as CL


def hot_image_db(mode):
    image_db = dict()

    if mode != 'shallow_1' and mode != 'shallow_2':
        return None

    data_url = 'static/reduce/' + mode + '_data.pkl'
    label_url = 'static/reduce/' + mode + '_label.pkl'
    f = open(data_url, 'r')
    image_db['data'] = pickle.load(f)
    f.close()
    f = open(label_url, 'r')
    image_db['label'] = pickle.load(f)
    f.close()
    return image_db


def hot_preprocesser():
    preprocesser = PP.vgg_bottom()
    return preprocesser


def hot_reducer(mode):
    reducer = RD.reducer_wrapper(mode)
    return reducer


def url_index(url='static/caltech101'):
    index = []

    pathDir = os.listdir(url)
    for clsname in pathDir[1:]:
        clsPath = os.listdir(url + '/' + clsname)
        for picname in clsPath[:]:
            imgurl = url + '/' + clsname + '/' + picname
            index.append(imgurl)
    return index


def similarity(query, image_db):
    sim = np.dot(image_db['data'], query.T)
    print sim
    rank = np.argsort(-sim, axis=0)
    print rank
    print image_db['data']
    print sim[rank].T
    print image_db['label'][rank].T

    return rank


class retreiver(object):
    def __init__(self, mode):
        print 'here'
        print mode
        if mode == "1":
            mode = 'shallow_1'
        elif mode == "2":
            mode = 'shallow_2'
        elif mode == "3":
            mode = 'kmeans_shallow_1'
        elif mode == "4":
            mode = 'kmeans_shallow_2'
        else:
            mode = 'shallow_2'

        print mode
        self.mode = mode
        self.image_db = hot_image_db(mode[-9:])
        self.reducer = hot_reducer(mode[-9:])
        self.preprocesser = hot_preprocesser()
        self.index = url_index(url='static/caltech101')
        if mode[:6] == 'kmeans':
            centers = pickle.load(open('static/params/{}_centers.pkl'.format(self.mode), 'r'))
            labels = pickle.load(open('static/params/{}_labels.pkl'.format(self.mode), 'r'))
            self.cluster = CL.kmeans_indexer(self.image_db, centers=centers, labels=labels, base=mode[-9:])

    def mode_warpper(self, mode):
        return mode

    def pic2fullvec(self, pic_url):
        imgurl = pic_url
        img = cv2.imread(imgurl)
        im = np.array(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        return im

    def preprocess(self, fullvec):
        pre_vec = self.preprocesser.predict(np.expand_dims(fullvec, axis=0), batch_size=1)
        return pre_vec[0]

    def reduce(self, post_pp):
        query = self.reducer.predict(np.expand_dims(post_pp, axis=0))
        return query[0]

    def retrieval(self, pic_url):
        res_list = []

        # convert url to full pic vec
        fullvec = self.pic2fullvec(pic_url)
        post_pp = self.preprocess(fullvec)
        query = self.reduce(post_pp)

        query = np.expand_dims(query, axis=0)

        if self.mode[:6] == 'kmeans':
            rank = self.cluster.retrieval(query)
        else:
            sim = np.dot(self.image_db['data'], query.T)[:, 0]
            rank = np.argsort(-sim)
            rank = rank.T.tolist()
            sim = sim.tolist()

            sim = map(lambda x: x > 0.1 and x or 0, sim)
            res_nb = reduce(lambda x, y: y > 0 and x + 1 or x, sim)
            if res_nb == 0:
                res_nb = 10
            rank = rank[:res_nb]

        res_list = [self.index[i] for i in rank]

        return res_list

    def remode(self, mode):

        print 'here'
        print mode
        # mode = 'shallow_1'
        if mode == "1":
            mode = 'shallow_1'
        elif mode == "2":
            mode = 'shallow_2'
        elif mode == "3":
            mode = 'kmeans_shallow_1'
        elif mode == "4":
            mode = 'kmeans_shallow_2'
        else:
            mode = 'shallow_2'

        print mode

        self.image_db = hot_image_db(mode[-9:])
        self.reducer = hot_reducer(mode[-9:])
        self.mode = mode
        if mode[:6] == 'kmeans':
            centers = pickle.load(open('static/params/{}_centers.pkl'.format(self.mode), 'r'))
            labels = pickle.load(open('static/params/{}_labels.pkl'.format(self.mode), 'r'))
            self.cluster = CL.kmeans_indexer(self.image_db, centers=centers, labels=labels, base=mode[-9:])