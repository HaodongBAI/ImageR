# -*- coding:utf-8 -*-
#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import pywt
from numpy.core.multiarray import dtype
import copy
import cPickle as pickle



def get_feature_vec(pic_name):
    #print 'open pic...'
    image =Image.open(pic_name)

    img = np.array(image)
    #print 'transfer to luv...'
    im_luv = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    L = im_luv[:, :, 0]  # get L
    LL, (HL, LH, HH) = pywt.dwt2(L, 'db4')  # get ll,lh,hl,hh
    width, height = image.size
    #print 'compute features...'
    x = 0
    feature_vec = []
    while((x + 4) < height):
        y = 0
        while((y + 4) < width):
            r = img[x:x + 4, y:y + 4, 0].sum() / 16
            g = img[x:x + 4, y:y + 4, 1].sum() / 16
            b = img[x:x + 4, y:y + 4, 2].sum() / 16
            mat = LH[x + 2:x + 4, y + 2:y + 4]
            v0 = mat * mat
            v0 = v0.sum() / 2
            mat = HL[x + 2:x + 4, y + 2:y + 4]
            v1 = mat * mat
            v1 = v1.sum() / 2
            mat = HH[x + 2:x + 4, y + 2:y + 4]
            v2 = mat * mat
            v2 = v2.sum() / 2
            feature_vec.append([r, g, b, v0, v1, v2])
            y += 4
        x += 4
    return feature_vec

def get_district(feature_vec, k=5):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(feature_vec)
    label = kmeans.labels_
    centers = kmeans.cluster_centers_
    districts = []
    while(k > 0):
        districts.append([])
        k -= 1
    i = 0
    while(i < len(label)):
        districts[label[i]].append(feature_vec[i])  # create districts by class
        i += 1
    return districts, centers  # [[feature_vec,..,],[],[]],centers

def g(d):
    return 1 if d >= 0.5 else 0.85 if d > 0.2 else 0.5

def get_shape_feature(districts, centers):
    shape_features = []  # [district1-[f7,f8,f9],...,]
    for (class_distr, clas) in zip(districts, centers):
        distri_shpe_featur = []  # [f7,f8,f9]
        for exp in [1, 2, 3]:
            numerator = 0
            denominator = pow(len(class_distr), 1 + exp / 6.0)
            for x in class_distr:  # x is feature_vec
                for element, center in zip(x, clas):
                    numerator += pow(abs(element - center), exp)
            distri_shpe_featur.append(numerator ** (1.0 / exp) / denominator)
        shape_features.append(distri_shpe_featur)
    normalze_deno = []  # normalize
    for i in [0, 1, 2]:
        normalze_deno.append(max(np.array(shape_features)[:, i]))
    p = 0
    for i in shape_features:
        q = 0
        for j in i:
            shape_features[p][q] /= normalze_deno[q]
            q += 1
        p += 1
    return shape_features  # [district1-[f7,f8,f9],...,]

    
def get_distance(p1_f1_6,pic1_sha_vec,p2_f1_6,pic2_sha_vec):
    distance_array = []
    p = 0
    for i, j in zip(p1_f1_6 , pic1_sha_vec):
        distance_array.append([])
        for x, y in zip(p2_f1_6, pic2_sha_vec):
            distance_array[p].append(get_g_ds((i + j)[6:9], (x + y)[6:9]) * get_dt((i + j)[:6], (x + y)[:6]))
        p += 1
    return distance_array

def get_p(districts):
    total = 0
    plist = []
    for i in districts:
        total += len(i)
    for i in districts:
        plist.append(len(i) / float(total))
    return plist
        
        
def get_sig_arr(distance_array, p1_p, p2_p):
    sig_arr = np.zeros((len(p1_p), len(p2_p)), dtype=float)
    while(sum(p1_p + p2_p) > 0.1e-10):
        i = 0
        while(i < len(p1_p)):
            pos = distance_array[i].index(min(distance_array[i]))
            if(p1_p[i] == 0.) | (p2_p[pos] == 0.):
                distance_array[i][pos] = max(distance_array[i]) + 1
                i+= 1
                continue
            if p1_p[i] > p2_p[pos]:
                sig_arr[i][pos] = p2_p[pos]
                p1_p[i] -= p2_p[pos]
                p2_p[pos] = 0
            else:
                sig_arr[i][pos] = p1_p[i]
                p2_p[pos] -= p1_p[i]
                p1_p[i] = 0
            distance_array[i][pos] = max(distance_array[i]) + 1
            i += 1
    return sig_arr

def get_sig_arr2(distance_array, p1_p, p2_p):
    sig_arr = np.zeros((len(p1_p), len(p2_p)), dtype=float)
    MAX_NUM=9999
    mini=-1
    while(sum(p1_p)> 0.1e-8 and sum(p2_p)>0.1e-8):
        mini=np.amin(distance_array)
        for i in range(len(p1_p)):
            for j in range(len(p2_p)):
                if(mini==distance_array[i][j]):
                    if p1_p[i]<p2_p[j]:
                        for k in range(len(p2_p)):
                            distance_array[i][k]=MAX_NUM
                    else:
                        for k in range(len(p1_p)):
                            distance_array[k][j]=MAX_NUM
                    sig_arr[i][j]=min(p1_p[i],p2_p[j])
                    p1_p[i]-=sig_arr[i][j]
                    p2_p[j]-=sig_arr[i][j]
                    distance_array[i][j]=MAX_NUM
                    mini=-1
                    break
            if -1==mini:
                break
    return sig_arr

def get_dt(p1_f1_6, p2_f1_6):
    dt = 0
    for i, j in zip(p1_f1_6, p2_f1_6):
        dt += (i - j) ** 2
    return dt

def get_g_ds(p1_f7_9, p2_f7_9):
    ds = 0
    for i, j in zip(p1_f7_9, p2_f7_9):
        ds += (i - j) ** 2
    return g(ds)

#pic1 = "E:\MT_huang\HW3\\dog1.jpg"
#pic2 = "E:\MT_huang\HW3\\bear.jpg"
#pic1_feature_vec = get_feature_vec(pic1)
#pic2_feature_vec = get_feature_vec(pic2)


#这是定义特征函数的返回类
class features():
    pp=[]
    pf6=[]
    pshape=[]

#这是由输入的图像获得特征函数的函数
def getFeature(pic):
    pic_temp=features()
    pic_feature_vec = get_feature_vec(pic)
    pic_distr,center=get_district(pic_feature_vec)
    pic_temp.pf6 = []
    k = 0
    for i in pic_distr:
        pic_temp.pf6.append([])
        j = 0
        while (j < 6):
            pic_temp.pf6[k].append(np.array(i)[:, j].sum() / len(i) / 255.0)
            j += 1
        k += 1
    pic_temp.pp = get_p(pic_distr)
    pic_temp.pshape = get_shape_feature(pic_distr, center)
    return pic_temp

#这是由两个向量获得距离的函数
def getDistance(temp1,temp2):
    pfea1=copy.deepcopy(temp1)
    pfea2=copy.deepcopy(temp2)
    distance = get_distance(pfea1.pf6,pfea1.pshape,pfea2.pf6,pfea2.pshape)
    distance_copy = copy.deepcopy(distance)
    #distance_copy=get_distance(pfea1.pf6, pfea1.pshape, pfea2.pf6, pfea2.pshape)
    sig_arr = get_sig_arr(distance_copy, pfea1.pp, pfea2.pp)
    dist = (np.array(distance) * np.array(sig_arr)).sum()
    return dist









