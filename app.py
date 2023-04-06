import io
from itertools import count
import os
import cv2
import flask
import numpy as np
from PIL import Image
from tabledef import *
from datetime import timedelta
from flask import request, session, jsonify, redirect
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
import face_recognition
import imutils
import shutil
import base64
from flask_cors import CORS
import requests
import json
from id_card_cropping.id_card_detection_image import cropping_id_card

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-------Connect to Database------#
engine = create_engine('sqlite:///login_db.db', echo=True)

#-------Import our model from folder-------#
from anti_spoofing.face_anti_spoofing import detect
from id_card_recognition.sift_flann import sift, match
from id_card_recognition.utils import findFaces

application = flask.Flask(__name__)
application.secret_key = 'web_app_for_face_recognition_and_liveness' # something super secret

CORS(application)

flg = 0
questionA = ""
image_array = []
number_question = 0
img_temple = []

#------Password Validate------#
def password_check(passwd):
    if len(passwd) < 6:
        return 'la longitud debe ser de al menos 6'
         
    if len(passwd) > 20:
        return 'la longitud no debe ser superior a 20'
         
    if not any(char.isdigit() for char in passwd):
        return 'La contraseña debe tener al menos un número'
         
    if not any(char.isupper() for char in passwd):
        return 'La contraseña debe tener al menos una letra mayúscula'
         
    if not any(char.islower() for char in passwd):
        return 'La contraseña debe tener al menos una letra minúscula'
         
    return 'success'


def face_recog(name, img):
    known_image = face_recognition.load_image_file("static/dataset/" + name + "/crop_face.png")
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    print('face_encoding')
    unknown_image = img
    face_encodings = face_recognition.face_encodings(unknown_image)
    face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]
    print(face_distance)

    Session = sessionmaker(bind=engine)
    s = Session()
    results = s.query(User).all()

    frame = img    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750) # scale down for faster process
    print('[INFO] recognizing faces...')
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    minn = 1
    for result in results:
        if result.encoded_face is None:
            continue        
        data = result.encoded_face
        dist = face_recognition.face_distance(encodings, data['encodings'][0])[0]
        # loop over the encoded faces
        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'Unknown'
            
            if True in matches:
                matchedIdxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                
                for i in matchedIdxs:
                    name = data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

        if(name != 'Unknown'):
            if(dist < minn):
                minn = dist
    res = "fail"
    if minn >= 0.4:
        res = "pass"
    if face_distance < 0.6:
        return "pass", res
    else:
        return "fail", res
    


#---------Recognize user's face----------
def recognize(img, useremail):
    Session = sessionmaker(bind=engine)
    s = Session()
    result = s.query(User).filter_by(email = useremail).first()

    print(result, useremail)
    frame = img    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750) # scale down for faster process
    print('[INFO] recognizing faces...')
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    minn = 1
    if not result.encoded_face is None:
        data = result.encoded_face
        dist = face_recognition.face_distance(encodings, data['encodings'][0])[0]
        # loop over the encoded faces
        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'Unknown'
            
            if True in matches:
                matchedIdxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                
                for i in matchedIdxs:
                    name = data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

        if(name != 'Unknown'):
            if(dist < minn):
                minn = dist

    print(minn)
    if(minn > 0.4): 
        return "fail"
    else :
        return "pass"

#---------Encode user's face----------
def encode_face(img, name, email, detection_method):
    knownEncodings = list()
    knownNames = list()

    rfc = session['rfc']
    image = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model = detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        # add each encoding and name to the list
        knownEncodings.append(encoding)
        knownNames.append(name)

    data = {'encodings': knownEncodings, 'names': knownNames}    
    Session = sessionmaker(bind=engine)
    s = Session()
    user = User(name, email, rfc, data)
    s.add(user)
    s.commit()
    
#------Timing out the login session------#

@application.before_request
def make_session_permanent():
    if session.get('username'):
        name = session['username']
    session.permanent = True
    application.permanent_session_lifetime = timedelta(minutes=8)

#------Index------#
@application.route('/')
def index():
    return flask.render_template("index.html")

#------IDCard Verification-------#
@application.route('/id_card')
def id_card():
    if not session.get('logged_in'):
        return index()
    else:
        return flask.render_template("id_card.html")

@application.route('/id_verification', methods = ["POST"])
def id_verification():

    data = {'success': False, 'face': True}
    if flask.request.method == "POST":
        direction = str(request.form['direction'])
        if flask.request.files.get("image"):
            name = session['username']
            img = flask.request.files["image"].read()
            img = np.array(Image.open(io.BytesIO(img)))
            dirName = "static/dataset/" + name

            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory " , dirName ,  " Created ")

            plt.imsave("static/dataset/" + name + "/id_card.png", img)

            if direction == 'front':
                is_id_card = cropping_id_card(name, 0) 
                face = sift(img)
                if is_id_card == False:
                    data['success'] = False
                elif face is None:
                    data['success'] = True
                    data['face'] = False
                else:
                    data['success'] = True
                    plt.imsave("static/dataset/" + name + "/crop_face.png", face)
            elif direction == 'back':
                is_id_card = cropping_id_card(name, 1) 
                face = sift(img)
                if is_id_card == False:
                    data['success'] = False
                elif not face is None:
                    data['success'] = True
                    data['face'] = False
                else:
                    session['id_capture'] = True
                    data['success'] = True
    return flask.jsonify(data)

#------Liveness And Face Recognition-------#
@application.route('/signature')
def signature():
    if not session.get('id_capture'):
        return id_card()
    else:
        return flask.render_template("signature.html")
    # return flask.render_template("signature.html")

flg = 0
user_img = ""

@application.route("/predict", methods=["POST"])
def predict():
    
    data = {'success': False,
            'id_ver': False,
            'is_cal': False,
            'is_start': False,
            'final': False,
            'not_find_face':False,
            'message': "OK",
            'check_user_face': True}

    if flask.request.method == "POST":
       
        count = int(request.form['counting'])
        question = str(request.form['question'])
        latitude = str(request.form['latitude'])
        longitude = str(request.form['longitude'])
        deviceID = str(request.form['deviceID'])
        
        global number_question
        global image_array
        global questionA
        global user_img
        
        session['latitude'] = latitude
        session['longitude'] = longitude

        name = session['username']
        email = session['useremail']
        data1 = { 
            "base64frontidcardpicture": "", 
            # "base64backidcardpicture": "",
            "base64frontemployeepicture": "", 
            "token": "", 
            "gpslatitude": "",  
            "gpslongitude": "", 
            "ide": 0,
            "deviceID": ""
        }
        if flask.request.files.get("image"):
            img = flask.request.files["image"].read()
            img = np.array(Image.open(io.BytesIO(img)))

            flag = 0
            challenge_res = 'fail'
            if question == 'final_img':
                user_img = img
                img_find_face = sift(img)
                data['final'] = True
                if not img_find_face is None:
                    img1 = img
                    h, w, _ = img.shape
                    h /= 2
                    w /= 2
                    print("--------previous, face_recod")
                    result, result1 = face_recog(name, img1)
                    print("face_recognition", result, result1)
                    if result1 == "fail":
                        data["check_user_face"] = False
                    elif (result == "pass"):
                        plt.imsave("static/dataset/" + name + "/liveness_face.png", cv2.resize(img1, (int(w), int(h))))
                        data['id_ver'] = True
                    else :
                        session['id_capture'] = False
                else: data['not_find_face'] = True
                return flask.jsonify(data)
            elif number_question < 25:
                if number_question == 1:
                    questionA = question
                number_question += 1
                image_array.append(img)
            if number_question == 25:
                flag = 1
                number_question = 0
                challenge_res = detect(image_array, questionA)
                image_array = []

            if flag == 1:
                data['is_cal'] = True
                if challenge_res == 'pass':
                    data['success'] = True
                    if count == 1:
                        with open("static/dataset/" + name + "/liveness_face.png", "rb") as img_file:
                            my_string1 = base64.b64encode(img_file.read())
                        with open("static/dataset/" + name + "/front.png", "rb") as img_file:
                            my_string = base64.b64encode(img_file.read())
                        with open("static/dataset/" + name + "/back.png", "rb") as img_file:
                            my_string2 = base64.b64encode(img_file.read())
                        ide = session['useride']
                        token = session['usertoken']
                        data1["base64frontidcardpicture"] = my_string.decode()
                        data1["base64frontemployeepicture"] = my_string1.decode()
                        # data1["base64backidcardpicture"] = my_string2.decode()
                        data1["token"] =  token
                        data1["gpslatitude"] =  str(latitude)
                        data1["gpslongitude"] =  str(longitude)
                        data1["ide"] =  ide
                        data1["deviceID"] = deviceID
                        print(token, latitude, longitude, ide, deviceID)

                        headers = {'content-type': 'application/json'}
                        data1 = json.dumps(data1)
                        res = requests.post('https://www.fiscoclic.mx/Nomina/rh/FCRH_firmadigital/SRV_onboardUser/', headers = headers, data=data1)
                        print(res.text)
                        ressss = res.json()
                        if(ressss['status'] == "ERROR"):
                            data['message'] = ressss['messageprocessed']
                        else : 
                            data['message'] = "OK"
                            encode_face(user_img, name, email, 'hog')

    return flask.jsonify(data)

@application.route('/f_recognition')
def f_recognition():
    if not session.get('logged_in'):
        return flask.render_template("index.html")
    # if session['url'] == 'only_test':
    #     return redirect("https://www.fiscoclic.mx")
    return flask.render_template("face_recognition.html")

@application.route('/recognition', methods=["POST"])
def recognition():
    data = {'success': False,
            'is_cal': False,
            'is_start': False,
            'final': False,
            'name': "Unknown",
            'not_find_face': False}

    if flask.request.method == "POST":
       
        question = str(request.form['question'])
        latitude = str(request.form['latitude'])
        longitude = str(request.form['longitude'])
        session['latitude'] = latitude
        session['longitude'] = longitude
        global flg
        global number_question
        global image_array
        global questionA
        
        if flask.request.files.get("image"):
          
            img = flask.request.files["image"].read()
            img = np.array(Image.open(io.BytesIO(img)))

            flag = 0
            if question == 'final_img':
                if flg ==0:
                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_find_face = findFaces(img1)
                    data['final'] = True
                    flg = 1
                    if not img_find_face is None:
                        name = session['username']
                        email = session['useremail']
                        result = recognize(img, email)
                        if result == 'pass':
                            data['name'] = name
                        else :
                            session['id_capture'] = False
                    else: data['not_find_face'] = True
                    return flask.jsonify(data)
            if number_question < 25:
                flg = 0
                if number_question == 1:
                    questionA = question
                number_question += 1
                image_array.append(img)
            challenge_res = 'fail'
            if number_question == 25:
                flag = 1
                number_question = 0
                challenge_res = detect(image_array, questionA)
                image_array = []

            if flag == 1:
                data['is_cal'] = True
                if challenge_res == 'pass':
                    data['success'] = True
    return flask.jsonify(data)

#------------api---------------
@application.route('/redirector', methods=["POST"])
def redirector():
    if flask.request.method == "POST" and 'url' in request.form and 'name' in request.form and 'email' in request.form and 'token' in request.form and 'ide' in request.form:
        status = str(request.form["status"])
        ide = int(request.form["ide"])
        if ide > 0:
            rfc = str(request.form["RFC"])
            url = str(request.form["url"])
            name = str(request.form["name"])
            email = str(request.form["email"])
            token = str(request.form["token"])
            ide = int(request.form['ide'])
            session['logged_in'] = True
            session['username'] = name
            session['useremail'] = email
            session['usertoken'] = token
            session['useride'] = ide
            session['url'] = url
            session['rfc'] = rfc
            if status == "OK":
                return f_recognition()
            else: return id_card()
    return index()
   
#----------clear_pic-----------#
@application.route('/clear_pic')
def clear_pic():
    print("---------clear_pic--------")
    url = session['url']
    name = session['username']
    token = session['usertoken']
    latitude = session['latitude']
    longitude = session['longitude']
    if url != "only_test" :
        url = url + '?token=' + token + '&gpslatitude=' + latitude + '&gpslongitude=' + longitude 
    if os.path.isdir('static/dataset/' + name):
        shutil.rmtree('static/dataset/' + name)
    session['logged_in'] = False
    session['id_capture'] = False
    session['username'] = ''
    session['useremail'] = ''
    session['usertoken'] = ''
    session['latitude'] = ''
    session['longitude'] = ''
    session['url'] = ''
    session['rfc'] = ''
    session['useride'] = 0
    global flg
    flg = 0
    print(url)
    if url == "only_test":
        return redirect("https://www.fiscoclic.mx")
    else :
        return flask.render_template("wait.html",  re_url = url)


#-----------delete_all------------
@application.route('/delete_all')
def delete_all():
    Session = sessionmaker(bind=engine)
    s = Session()
    s.query(User).delete()
    s.commit()
    session['logged_in'] = False
    session['id_capture'] = False
    session['username'] = ''
    session['useremail'] = ''
    session['usertoken'] = ''
    session['rfc'] = ''
    session['useride'] = 0
    if os.path.isdir('static/dataset/'):
        shutil.rmtree('static/dataset/')
    global flg
    flg = 0
    return index()

@application.route('/deleteByID/<RFC>')
def deleteByID(RFC):
    password = request.args.get('p', default = "", type = str)
    if password != "Carlos":
        return "ERROR"
    Session = sessionmaker(bind=engine)
    s = Session()
    result = s.query(User).filter_by(rfc = RFC).first()
    if(result is None):
        return "ERROR"
    session['logged_in'] = False
    session['id_capture'] = False
    session['username'] = ''
    session['useremail'] = ''
    session['usertoken'] = ''
    session['rfc'] = ''
    session['useride'] = 0
    global flg
    flg = 0
    s.query(User).filter_by(rfc = RFC).delete()
    s.commit()
    return "OK"
    

if __name__ == "__main__":

    print("** Starting Flask server.........Please wait until the server starts ")
    print('Loading the Neural Network......\n')

    application.run(host = '0.0.0.0', port = '8080', debug=True)
    
  
