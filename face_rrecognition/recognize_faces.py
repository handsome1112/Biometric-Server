import face_recognition
import pickle
import cv2
import imutils

def recognize(img):

    print('[INFO] loading encodings...')
    with open('face_rrecognition/encoded_faces.pickle', 'rb') as file:
        data = pickle.loads(file.read())

    frame = img
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750) # scale down for faster process
    r = frame.shape[1] / float(rgb.shape[1]) # get the scale ratio for later use in puting text
    print('[INFO] recognizing faces...')
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    
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
            
        return name