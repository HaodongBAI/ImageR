# -*- coding:utf-8 -*-
from lib import *


@app.route('/',methods=['GET','POST'])
def home2():
    #print os.getcwd()
    if  request.method == 'POST':
        photo = request.files['file']
        photo.save(os.path.join(UPLOAD_FOLDER,photo.filename))
        pic_now.picname = photo.filename
        pic_now.picmodel = request.form['model']
        print pic_now.picmodel
        pic_now.picurl = ('static/upload/'+pic_now.picname)
        ptime,pclass=getlist(pic_now.picurl, pic_now.picmodel)
        session['ptime']=ptime
        session['pclass']=pclass
        return redirect(url_for('home2'))
    return render_template('home2.html',plist=pic_now.piclist,
                           pname=pic_now.picname,pmodel=pic_now.picmodel,
                           plen=len(pic_now.piclist),purl=pic_now.picurl,
                           ptime=session.get('ptime'),pclass=session.get('pclass'))


@app.route('/search',methods=['GET', 'POST'])
def search():
    query = wordQuery()
    if query.validate_on_submit():
        session['name'] = query.name.data
        query.name.data=''
        return redirect(url_for('serach'))
    return render_template('search.html', query=query, name=session.get('name'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/base')
def base():
    return render_template('base.html')


@app.route('/gallery')
def gallery():
    return render_template('gallery.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run()
