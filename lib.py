# -*- coding:utf-8 -*-
from flask import Flask, render_template
from flask import session, redirect, url_for, flash, request
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import required
from werkzeug import security
import os
from flask_sqlalchemy import _SQLAlchemyState
import datetime
from retrieval import retreiver, url_index
from irm import *
import timeit
import reducer as RD

app = Flask(__name__)
os.chdir('/Users/Kevin/Documents/Develop/PyCharmPython/nova')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['SECRET_KEY'] = 'this is a string--LR328RHEWFILJR21OR1E12NR309R0FN'

bootstrap = Bootstrap(app)
moment = Moment(app)


class wordQuery(FlaskForm):
    name = StringField('', validators=[required()])
    submit = SubmitField('Submit')

class pic():
    picname = ''
    picmodel = ''
    piclist = []
    picurl = ''
    for i in range(1, 25):
        piclist.append('static/images/pic' + str(i) + '.jpg')

pic_now = pic()
rtr = retreiver(pic_now.picmodel)

piclist = url_index()
length = len(piclist)
fealist = pickle.load(file('static/params/irm.pkl', 'r'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# !!!这是由输入图片向结果列表变化的函数
def getlist(pic,pmodel):
    print type(pmodel)
    start_time = timeit.default_timer()
    if("0"==pmodel):
        quefea=getFeature(pic)
        distlist=length*[0]
        templist=100*['0']
        for j in range(length):
            if (fealist[j].pp == [0, 0, 0, 0, 0]):
                temp = 9999
            else:
                temp = getDistance(quefea, fealist[j])
            distlist[j] = temp
        discopy = copy.deepcopy(distlist)
        distlist.sort()
        distlist=distlist[0:100]
        for j in range(100):
            temp = piclist[discopy.index(distlist[j])]
            templist[j]=temp
        pic_now.piclist=templist
    else:
        rtr.remode(pmodel)
        pic_now.piclist = rtr.retrieval(pic)
    end_time = timeit.default_timer()
    temp=pic_now.piclist[0]
    beg = temp.find('ch101')
    end=temp.find('image')
    temp=temp[beg+6:end-1]
    pclass=temp
    return int(1000*(end_time*1.0 - start_time)),pclass


# 获得部分query的url集合

'''
templist = pickle.load(file('static/params/class.pkl', 'r'))
        classdict = {}
        for i in templist:
            classdict[i[0]] = i[1]
        quelen=len(querylist)
        result=[]
        for i in range(quelen):
            print i, '/', quelen,querylist[i]
            distlist=length*[0]
            classlist=length*[0]
            try:
                query_fea = getFeature(querylist[i])

                for j in range(length):
                    if(fealist[j].pp==[0,0,0,0,0]):
                        temp=9999
                    else:
                        temp=getDistance(query_fea,fealist[j])
                    distlist[j]=temp
                discopy=copy.deepcopy(distlist)
                distlist.sort()
                for j in range(length):
                    if 9999==distlist[j]:
                        classlist[j]=0
                    else:
                        temp=piclist[discopy.index(distlist[j])]
                        beg=temp.find('ch101')
                        end=temp.find('image')
                        temp=temp[beg+6:end-1]
                        classlist[j]=classdict[temp]

                result.append(classlist)
                print classlist[0:100]
                print '\n'
            except Exception:
                result.append(classlist)
                print "Error!"
            if 0==i%10:
                pickle.dump(result, file('static/params/irm_res.pkl', 'w'))

# templist = pickle.load(file('static/params/class.pkl', 'r'))
# classdict = {}
# for i in templist:
#     classdict[i[0]] = i[1]
#
# query_url = url_index()
# image_db = RD.load_pre_caltech()
# sub_idx = RD.subset(image_db, r=1)
# querylist = [query_url[i] for i in sub_idx]
#
# piclist = url_index()
# print len(piclist)
# length = len(piclist)
#
# fealist = pickle.load(file('static/params/irm.pkl', 'r'))
#
# quelen=len(querylist)
# result=[]
# for i in range(quelen):
#     print i, '/', quelen,querylist[i]
#     distlist=length*[0]
#     classlist=length*[0]
#     try:
#         query_fea = getFeature(querylist[i])
#
#         for j in range(length):
#             if(fealist[j].pp==[0,0,0,0,0]):
#                 temp=9999
#             else:
#                 temp=getDistance(query_fea,fealist[j])
#             distlist[j]=temp
#         discopy=copy.deepcopy(distlist)
#         distlist.sort()
#         for j in range(length):
#             if 9999==distlist[j]:
#                 classlist[j]=0
#             else:
#                 temp=piclist[discopy.index(distlist[j])]
#                 beg=temp.find('ch101')
#                 end=temp.find('image')
#                 temp=temp[beg+6:end-1]
#                 classlist[j]=classdict[temp]
#
#         result.append(classlist)
#         print classlist[0:100]
#         print '\n'
#     except Exception:
#         result.append(classlist)
#         print "Error!"
#     if 0==i%10:
#         pickle.dump(result, file('static/params/irm_res.pkl', 'w'))
#
#
#
# pickle.dump(result, file('static/params/irm_res.pkl', 'w'))
# tempresult=pickle.load(file('static/params/irm_res.pkl', 'r'))
# print tempresult
'''



'''url= 'static/caltech101'
class_dict = dict()
pathDir = os.listdir(url)
for clsname in pathDir[1:]:
    label = len(class_dict)
    class_dict[clsname] = label+1

templist = pickle.load(file('static/params/class.pkl', 'r'))
classdict = {}
for i in templist:
    classdict[i[0]] = i[1]
print classdict


'''
'''
length = len(fealist)
distlist = length * [0]
classlist = length * ['0']
query = fealist[0]
for i in range(length):
    print i,
    if (fealist[i].pp == [0, 0, 0, 0, 0]):
        temp = 9999
    else:
        temp = getDistance(query, fealist[i])
    distlist[i] = temp
    print temp

discopy = copy.deepcopy(distlist)
distlist.sort()
for i in range(length):
    if 9999 == distlist[i]:
        classlist[i] = 'NULL'
    else:
        classlist[i] = piclist[discopy.index(distlist[i])]
        temp = classlist[i]
        beg = temp.find('ch101')
        end = temp.find('image')
        temp = temp[beg + 7:end - 1]
        classlist[i] = temp

print distlist
print classlist
classdict={}
k=1
for i in range(length):
    temp=piclist[i]
    beg=temp.find('ch101')
    end=temp.find('image')
    temp=temp[beg+6:end-1]
    if not classdict.has_key(temp):
        classdict[temp]=k
        print temp
        k+=1
classdict=sorted(classdict.iteritems(),key=lambda d:d[0])

classdict=sorted(classdict.iteritems(),key=lambda d:d[0])
print classdict
pickle.dump(classdict, file('static/params/class.pkl', 'w'))

'''

'''
pfea1.pp=5*[0]
pfea1.pf6=5*[[0,0,0,0,0,0]]
pfea1.pshape=5*[[0,0,0]]

piclist=url_index()
fealist=[]
k=0
for i in range(len(piclist)):
    print k,
    k += 1
    print piclist[i]
    try:
        temp=getFeature(piclist[i])
    except Exception:
        temp=pfea1
    if(9==i):
        print temp.pf6
    fealist.append(temp)

pickle.dump(fealist, file('static/params/irm.pkl', 'w'))
templist=pickle.load(file('static/params/irm.pkl', 'r'))
print len(templist)



pic1 = "static/caltech101/Faces/image_0001.jpg"
pfea1=getFeature(pic1)
fealist.append(pfea1)

for i in range(101,110):
    temp=getFeature(piclist[i])
    print piclist[i]
    dist=getDistance(pfea1,temp)
    fealist.append(temp)
    distlist.append(dist)
    print str(i)+":",
    print dist
'''

'''
class picQuery(FlaskForm):
    photo=FileField('Your Photo')
    model=SelectField('Model',choices=[('1','CNN-1'),('2','CNN-2'),('3','CNN-3'),('4','CNN-4'),('0','IRM')])
    submit=SubmitField('Upload')
@app.route('/home2',methods=['GET', 'POST'])
def home2():
    pic=picQuery()
    if pic.validate_on_submit():
        session['photo']=pic.photo.data.filename
        session['model']=pic.model.data
        pic.photo.data=None
        pic.model.data=''
        return redirect(url_for('home2'))
    return render_template('home2.html',pic=pic,photo=session.get('photo'),model=session.get('model'))
@app.route('/home', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        None
       # session.clear()
    elif request.method == 'POST':
        photo = request.files['file']
        session['pname'] = photo.filename
        photo.save(os.path.join(UPLOAD_FOLDER,photo.filename))
        session['pmodel'] = request.form.get('model','0')
        pic_now.piclist = getlist(photo)
        pic_now.picname = photo.filename
        pic_now.picmodel = request.form.get('model','0')
    return render_template('home.html',pname=session.get('pname'),pmodel=session.get('pmodel'))
       # return redirect(url_for('upload_file'))
    #return render_template('result.html',plist=pic_now.piclist,pname=pic_now.picname,pmodel=pic_now.picmodel,plen=len(pic_now.piclist))

@app.route('/result',methods=['GET', 'POST'])
def result():
    #return render_template('gallery2.html',plist=piclist,pic=session.get('pname'))
    return render_template('result.html',
                           plist=pic_now.piclist,
                           pname=pic_now.picname,
                           pmodel=pic_now.picmodel,
                           plen=len(pic_now.piclist))
'''
