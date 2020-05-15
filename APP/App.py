# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
from Locate import text_to_clip

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
@app.route('/<vname>', methods=['POST', 'GET'])
def index(vname=None):
    if request.method == 'POST':
            f = request.files['file']
            video_save_path = os.path.join('D:\\Data\\Text-to-Clip\\APP\\static\\video',secure_filename(f.filename)) 
            vname = secure_filename(f.filename)
            f.save(video_save_path)
            return redirect(url_for('index',vname=vname))
    if vname:
        return render_template('index.html',vname=vname)
    else:  
        return render_template('index.html')

@app.route('/getClips', methods=['GET'])
def getClips():
    vname = request.args.get("vname")
    sentence = request.args.get("sentence")
    print("----------------------------------------------")
    print(vname)
    print(sentence)
    clips = text_to_clip(vname,sentence)
    return render_template('clips.html',clips=clips)
if __name__ == '__main__':
    app.run(debug=True)