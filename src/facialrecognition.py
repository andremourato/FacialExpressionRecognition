import cv2
import sys
from label_image import *

size = 4

input_name = None

#Only program name provided
if len(sys.argv) == 1:
    input_name = 0 #Using default WebCam connected to the PC.
elif len(sys.argv) == 2:
    input_name = sys.argv[1]
else:
    print('Usage: python facialrecognition.py [input-video-or-image]')

image_input = cv2.VideoCapture(input_name) 

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
graph = load_graph("retrained_graph.pb")
# graph.finalize()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
# gpu_options.allow_growth=False

with tf.Session(graph=graph,
    config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    while True:
        (rval, im) = image_input.read()
        # Resize the image to speed up detection
        mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

        # gets all the faces
        faces = classifier.detectMultiScale(mini)
        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] 
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
            
            #Save just the rectangle faces in SubRecFaces
            zoomed_Face = im[y:y+h, x:x+w]

            CurrentFaceFile = "test.jpg" 
            cv2.imwrite(CurrentFaceFile, zoomed_Face)
            
            text = main(CurrentFaceFile,graph,sess)# Gets the corresponding label (angry,happy,etc.)
            text = text.title()
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(im, text,(x+w,y), font, 1, (0,0,255), 2)
        
        # Show the image
        cv2.imwrite('output.jpg', im)
        cv2.imshow('Capture',   im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key
            break
