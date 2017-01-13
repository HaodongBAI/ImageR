# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import numpy as np
import timeit
import cv2
import reducer as RD
import retrieval as RE
import preprocesser as PP
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cPickle as pickle


class kmeans_indexer(object):
    def __init__(self, image_db, algo='kmeans', base='shallow_1', labels=None, centers=None):
        self.base = base
        self.image_db = image_db
        if labels is None and centers is None:
            if algo == 'kmeans':
                self.labels, self.centers, self.blocks = self.do_kmeans(image_db['data'], 101)
                self.cls = 101
        else:
            self.cls = len(centers)
            self.labels = np.asarray(labels).reshape((len(labels), 1))
            self.centers = np.asarray(centers).reshape((self.cls, len(image_db['data'][0])))
            self.blocks = []
            for i in range(self.cls):
                self.blocks.append([])

            for i in range(len(self.labels)):
                self.blocks[self.labels[i]].append(i)

    def do_kmeans(self, image_db, cls=101):

        print 'Start kmeans...'
        kmeans = KMeans(n_clusters=cls)
        kmeans.fit(image_db)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        pickle.dump(centers, open('static/params/kmeans_{}_centers.pkl'.format(self.base), 'w'), -1)
        pickle.dump(labels, open('static/params/kmeans_{}_labels.pkl'.format(self.base), 'w'), -1)

        blocks = []
        for i in range(cls):
            blocks.append([])
        for i in range(len(labels)):
            blocks[labels[i]].append(i)

        return np.asarray(labels).reshape(len(labels), 1), np.asarray(centers).reshape(cls, len(image_db[0])), blocks

    def hierarchy_matching(self, query_db, maxcls=10):
        nb_pic = len(self.image_db['label'])
        nb_query = len(query_db['label'])

        sim1 = np.dot(query_db['data'], self.centers.T)

        rank1 = np.argsort(-sim1, axis=1)

        res_labels = np.zeros(shape=(nb_query, nb_pic))

        for q_i in range(nb_query):
            idx_to_match = []
            for lbs in rank1[q_i][:3]:
                idx_to_match.extend(self.blocks[lbs])

            sim2 = np.dot(query_db['data'][q_i], self.image_db['data'][idx_to_match].T)
            rank2 = np.argsort(-sim2)
            nb_relate = len(rank2)
            for i in range(nb_relate):
                res_labels[q_i][i] = self.image_db['label'][idx_to_match[i]][0]

        return res_labels

    def retrieval(self, query, maxcls=10):

        sim1 = np.dot(query, self.centers.T)

        rank1 = np.argsort(-sim1, axis=1)

        idx_to_match = []
        for lbs in rank1[0][:maxcls]:
            idx_to_match.extend(self.blocks[lbs])

        sim2 = np.dot(query, self.image_db['data'][idx_to_match].T)
        rank2 = np.argsort(-sim2)

        rank2 = rank2[0].T.tolist()
        sim2 = sim2[0].tolist()
        # print rank2
        # print sim2

        sim2 = map(lambda x: x > 0.1 and x or 0, sim2)
        res_nb=0
        for i in sim2:
            if i > 0:
                res_nb+=1
        # res_nb = reduce(lambda x, y: y > 0 and x + 1 or x, sim2)
        # print rank2
        # print sim2
        # print res_nb
        if res_nb == 0:
            res_nb = 10
        rank2 = rank2[:res_nb]

        return [idx_to_match[i] for i in rank2]
