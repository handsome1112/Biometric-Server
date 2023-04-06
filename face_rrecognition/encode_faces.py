import face_recognition
import pickle
import cv2
import os

print('[INFO] quantifying faces...')
def encode_face(img, name, email, detection_method):
    knownEncodings = list()
    knownNames = list()

    image = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    dirName = "face_rrecognition/train_model/" + email

    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    # print("ok")
    boxes = face_recognition.face_locations(rgb, model = detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        # add each encoding and name to the list
        knownEncodings.append(encoding)
        knownNames.append(name)

    data = {'encodings': knownEncodings, 'names': knownNames}    
    with open(dirName + '/encoded_faces.pickle', 'wb') as file:
        file.write(pickle.dumps(data))