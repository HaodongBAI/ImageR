# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import EigenvalueRegularizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import keras.backend as K
import numpy as np
import os
import timeit
import cPickle as pickle


def subset(image_db, r=10):
    nb_instance = len(image_db['label'])
    k = 1
    same_class = []
    res = []
    for i in range(nb_instance):
        if image_db['label'][i] == k:
            same_class.append(i)
        else:
            np.random.shuffle(same_class)
            for j in range(r):
                res.append(same_class[j])
            same_class = []
            same_class.append(i)
            k += 1

    np.random.shuffle(same_class)
    for j in range(r):
        res.append(same_class[j])
    return res


def load_pre_caltech():
    image_db = dict()
    image_db['label'] = []
    image_db['data'] = []

    for k in range(1, 102):
        # print k
        f = open('static/preprocess/pred_{}.pkl'.format(k), 'rb')
        data = pickle.load(f)
        image_db['data'].append(data[:, ])
        labels = np.full(shape=(len(data), 1), fill_value=k, dtype=np.int32)
        image_db['label'].append(labels)

    image_db['data'] = np.vstack(image_db['data'])
    image_db['label'] = np.vstack(image_db['label'])

    return image_db


def shallow_1(nb_epoch=5, batch_size=32):
    K.set_image_dim_ordering('th')

    whole_model_weights_path = 'static/params/shallow_1.h5'

    model = Sequential()
    model.add(Dense(101, activation='softmax', input_shape=(4096,), W_regularizer=EigenvalueRegularizer(10)))

    model.compile(loss="categorical_crossentropy",
                  optimizer='Adadelta',
                  metrics=['mean_squared_logarithmic_error', 'accuracy'])

    image_db = load_pre_caltech()

    train_subset = subset(image_db, 30)

    image_db['data'] = image_db['data'][train_subset]
    image_db['label'] = image_db['label'][train_subset]

    enc = OneHotEncoder()
    image_db['label'] = np.reshape(image_db['label'], (len(image_db['label']), 1))
    enc.fit(image_db['label'])
    image_db['label'] = np.asarray(enc.transform(image_db['label']).toarray())

    print 'Data loaded. Shape = ' + str(image_db['data'].shape) + str(
        image_db['label'].shape) + ' Then, start training...'

    start_time = timeit.default_timer()
    model.fit(x=image_db['data'], y=image_db['label'],
              batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True)

    model.save_weights(whole_model_weights_path)

    end_time = timeit.default_timer()
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))


def shallow_2(nb_epoch=3, batch_size=32):
    K.set_image_dim_ordering('th')
    whole_model_weights_path = 'static/params/shallow_2.h5'

    model = Sequential()
    model.add(Dense(1024, activation='tanh', input_shape=(4096,), W_regularizer=EigenvalueRegularizer(10)))
    model.add(Dense(101, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))

    model.compile(loss="categorical_crossentropy",
                  optimizer='Adadelta',
                  metrics=['mean_squared_logarithmic_error', 'accuracy'])

    image_db = load_pre_caltech()

    train_subset = subset(image_db, 30)

    image_db['data'] = image_db['data'][train_subset]
    image_db['label'] = image_db['label'][train_subset]

    enc = OneHotEncoder()
    image_db['label'] = np.reshape(image_db['label'], (len(image_db['label']), 1))
    enc.fit(image_db['label'])
    image_db['label'] = np.asarray(enc.transform(image_db['label']).toarray())

    print 'Data loaded. Shape = ' + str(image_db['data'].shape) + ' Then, start training...'

    start_time = timeit.default_timer()
    model.fit(x=image_db['data'], y=image_db['label'],
              batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True)

    model.save_weights(whole_model_weights_path)

    end_time = timeit.default_timer()
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))


def shallow_reduce(mode=1):
    K.set_image_dim_ordering('th')

    model = Sequential()

    if mode == 1:
        model.add(Dense(101, input_shape=(4096,), activation='softmax'))
        weights_path = 'static/params/shallow_1.h5'
    elif mode == 2:
        model.add(Dense(1024, activation='tanh', input_shape=(4096,)))
        model.add(Dense(101, activation='softmax'))
        weights_path = 'static/params/shallow_2.h5'
    else:
        return

    model.load_weights(weights_path)
    print('Model loaded. Start to predict')

    image_db = load_pre_caltech()

    X_low = model.predict(image_db['data'], batch_size=32)
    print X_low.shape
    print 'start to pack tsne...'
    fp = open('static/reduce/shallow_{}_data.pkl'.format(mode), 'wb')
    pickle.dump(X_low, fp, -1)
    fl = open('static/reduce/shallow_{}_label.pkl'.format(mode), 'wb')
    pickle.dump(image_db['label'], fl, -1)
    print '\tfinish packing pred...\n'


def reducer_wrapper(mode):
    if mode == 'shallow_1':
        K.set_image_dim_ordering('th')
        model = Sequential()
        model.add(Dense(101, input_shape=(4096,), activation='softmax'))
        weights_path = 'static/params/shallow_1.h5'
        model.load_weights(weights_path)
        print 'Reducer loaded...'

        # def reducer_shallow1(hvec):
        #     lvec = model.predict(hvec)
        #     return lvec

        return model

    elif mode == 'shallow_2':
        K.set_image_dim_ordering('th')
        model = Sequential()
        model.add(Dense(1024, activation='tanh', input_shape=(4096,)))
        model.add(Dense(101, activation='softmax'))
        weights_path = 'static/params/shallow_2.h5'
        model.load_weights(weights_path)
        print 'Reducer loaded...'

        # def reducer_shallow2(hvec):
        #     lvec = model.predict(hvec)
        #     return lvec

        return model
    else:
        return None
