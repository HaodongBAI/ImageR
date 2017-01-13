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
import cluster as CL
from ggplot import *
import cPickle as pickle


def preprocess_batch_wrapper():
    model = PP.vgg_bottom()

    def preprocess_query(query_url):
        imgurl = query_url
        img = cv2.imread(imgurl)
        im = np.array(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        pre_vec = model.predict(np.expand_dims(im, axis=0), batch_size=1)  # 4096
        return pre_vec[0]

    return preprocess_query


def experiment(mode):
    image_db = RE.hot_image_db(mode[-9:])
    query_db = RD.load_pre_caltech()
    reducer = RD.reducer_wrapper(mode[-9:])
    query_subset = RD.subset(image_db)
    query_db['data'] = query_db['data'][query_subset]
    query_db['label'] = query_db['label'][query_subset]

    search_start_time = timeit.default_timer()

    query_db['data'] = reducer.predict(query_db['data'])
    nb_query = len(query_db['label'])

    if mode == 'shallow_1' or mode == 'shallow_2':
        # 得到相似度排名
        sim = np.dot(query_db['data'], image_db['data'].T)
        rank = np.argsort(-sim, axis=1)
        res_label = image_db['label'][rank]
        res_label = res_label.reshape(res_label.shape[0], res_label.shape[1])

        search_end_time = timeit.default_timer()

    elif mode == 'kmeans_shallow_1' or mode == 'kmeans_shallow_2':

        centers = pickle.load(open('static/params/{}_centers.pkl'.format(mode), 'r'))
        labels = pickle.load(open('static/params/{}_labels.pkl'.format(mode), 'r'))

        print 'Building matcher...'
        matcher = CL.kmeans_indexer(image_db, centers=centers, labels=labels, base=mode[-9:])
        print 'Matcher built...'
        # pickle.dump(matcher, open('hierarchy.pkl', 'w'), -1)

        search_start_time = timeit.default_timer()
        res_label = matcher.hierarchy_matching(query_db, image_db)
        search_end_time = timeit.default_timer()
    else:
        return None, None, None


    print 'Start to calc p-r matrix...'
    # 得到P-R矩阵
    r = (np.arange(10) + 1) / 10.0
    pr_matrix = np.zeros(shape=(nb_query, 10))
    relate = np.zeros(shape=(nb_query, len(image_db['label'])))
    for i in range(nb_query):
        relate[i] = (res_label[i] == query_db['label'][i])
    nb_relate = np.sum(relate, axis=1)
    relate_cumsum = np.cumsum(relate, axis=1)
    recall_lv = np.ceil(np.dot(np.expand_dims(nb_relate, axis=1), np.expand_dims(r, axis=0)))
    for i in range(nb_query):
        k = 0
        for j in range(len(relate_cumsum[i])):
            if relate_cumsum[i][j] == recall_lv[i][k]:
                pr_matrix[i][k] = recall_lv[i][k] / (1.0 * (j + 1))
                k += 1
                if k == 10:
                    break
    print 'pr matrix'
    # for i in pr_matrix:
    #     for j in i:
    #         print j,
    #     print '\n'

    # 得到MAP矩阵
    ap_vec = np.zeros(shape=(nb_query,))
    for i in range(nb_query):
        k = 1
        for j in range(len(relate_cumsum[i])):
            if relate_cumsum[i][j] == k:
                ap_vec[i] += k / (1.0 * (j + 1))
                k += 1
                if k > nb_relate[i]:
                    ap_vec[i] /= nb_relate[i]
                    break
    print 'ap vector'
    # print ap_vec

    # 得到时间向量
    search_time = search_end_time - search_start_time

    # 返回p-r矩阵,map矩阵,时间向量
    return pr_matrix, ap_vec, search_time


def visualize():
    modes = ['shallow_1', 'shallow_2', 'kmeans_shallow_1', 'kmeans_shallow_2']
    mpr = []
    mAP = []
    time = []
    for mode in modes:
        pr, ap, ti = experiment(mode)
        mpr.append(np.mean(pr, axis=0))
        mAP.append(np.mean(ap, axis=0))
        time.append(ti)

    plt.figure()
    r_lv = (np.arange(10) + 1) / 10.0
    lg0, = plt.plot(r_lv, mpr[0], '-', color='red')
    lg1, = plt.plot(r_lv, mpr[1], '--', color='blue')
    lg2, = plt.plot(r_lv, mpr[2], '-', color='green')
    lg3, = plt.plot(r_lv, mpr[3], '--', color='black')
    plt.xlim(0.1, 1.0)
    plt.ylim(0, 1.0)
    plt.legend([lg0, lg1, lg2, lg3], modes, loc=0, borderaxespad=0.)
    plt.show()

    print mAP
    print time







