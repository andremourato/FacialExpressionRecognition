import cv2
import sys
from random import randint

emotions = ['Angry','Sad','Calm','Happy']

def train_emotion_from_webcam(person_name,emotion,MAX_PHOTOS=50):
    # We load the xml file
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.
    num_photos_taken = 0
    size = 4
    while num_photos_taken < MAX_PHOTOS:
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,0) #Flip to act as a mirror
        # Resize the image to speed up detection
        mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))
        # detect faces in photo 
        faces = classifier.detectMultiScale(mini)
        # Draw green rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #SCALE THE COPY
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
            #Save the cropped image
            zoomed_Face = im[y:y+h, x:x+w]
            CurrentFaceFile = "./images/%s/%s_%d.jpg"%(emotion,person_name,randint(0,100000000))
            cv2.imwrite(CurrentFaceFile, zoomed_Face)
            num_photos_taken += 1
            # Show the image
        cv2.imshow('Capture',   im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key
            break

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python train_emotion.py <person-name> <emotion>')
        exit(1)
    if sys.argv[2] not in emotions:
        print('Emotion %s is not supported'%(sys.argv[2]))
        exit(1)
    train_emotion_from_webcam(sys.argv[1],sys.argv[2])