import cv2
from label_image import *

size = 4


# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.

graph = load_graph("retrained_graph.pb")

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)

with tf.Session(graph=graph,
    config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    while True:
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,0) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(mini)
        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
            
            #Save just the rectangle faces in SubRecFaces
            zoomed_Face = im[y:y+h, x:x+w]

            CurrentFaceFile = "test.jpg" #Saving the current image from the webcam for testing.
            cv2.imwrite(CurrentFaceFile, zoomed_Face)
            
            text = main(CurrentFaceFile,graph,sess)# Gets the corresponding label (angry,happy,etc.)
            text = text.title()
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(im, text,(x+w,y), font, 1, (0,0,255), 2)

        # Show the image
        cv2.imshow('Capture',   im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key
            break
