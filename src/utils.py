import cv2
import sys
import os
from random import randint

emotions = ['Angry','Sad','Calm','Happy','Surprise']
main_directory = './images/'


def facecrop(image):
    ## Crops the face of a person from any image!

    ## OpenCV XML FILE for Frontal Facial Detection using HAAR CASCADES.
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    ## Reading the given Image with OpenCV
    img = cv2.imread(image)

    try:
        ## Some downloaded images are of unsupported type and should be ignored while raising Exception, so for that
        ## I'm using the try/except functions.

        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)
        print('Found ',len(faces),' faces')
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]
            
            f_name = image.split('/')
            f_name = f_name[-1]
            ## Change here the Desired directory.
            # new_file_name = os.path.join(main_directory,emotion,f_name)
            cv2.imwrite(image, sub_face)
            print ("Writing: " + image)
    except:
        pass

def format_faces(face_images=None):
    if not face_images:
        emotions = os.listdir(main_directory)
        for emotion in emotions:
            emotion_path = os.path.join(main_directory,emotion)
            print(emotion_path)
            print(os.path.exists(emotion_path))

            print(emotion_path)
            images = os.listdir(emotion_path)
            i = 0
            for img in images:
                file = os.path.join(emotion_path,img)
                print ('Cropping ',file)
                facecrop(file)
                i += 1   
    else:
        for img in face_images:
            print ('Cropping ',img)
            facecrop(img)

def train_emotion_from_webcam(person_name,emotion,MAX_PHOTOS=50):
    # We load the xml file
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.
    num_photos_taken = 0
    size = 4
    new_files_created = []
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
            new_files_created += [CurrentFaceFile]
            # Show the image
        cv2.imshow('Capture',   im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key
            break
    format_faces(new_files_created)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python utils.py <function>')
        exit(1)
    #Usage: python utils.py train_emotion_from_webcam <person-name> <emotion>
    if sys.argv[1] == 'train_emotion_from_webcam':
        if len(sys.argv) < 4:
            print('Usage: python utils.py train_emotion_from_webcam <person-name> <emotion>')
            exit(1)
        emotion_path = os.path.join(main_directory,sys.argv[3])
        if not os.path.exists(emotion_path):
            print('New emotion path created!')
            os.mkdir(emotion_path)
        if sys.argv[3] not in emotions:
            print('Emotion %s is not supported'%(sys.argv[3]))
            exit(1)
        train_emotion_from_webcam(sys.argv[2],sys.argv[3])
    #Usage: python utils.py format_faces
    elif sys.argv[1] == 'format_faces':
        format_faces()
    else:
        print('Function ',sys.argv[1],' is not supported!')
        exit(1)