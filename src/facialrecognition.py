import cv2
import sys
from label_image import *
import datetime

size = 4
supported_image_types = ['png','jpg','jpeg']
is_image = False

image_input = None
#Only program name provided
if len(sys.argv) == 1:
    image_input = cv2.VideoCapture(0) #Using default WebCam connected to the PC.
elif len(sys.argv) == 2:
    if any([sys.argv[1].endswith('.'+ext) for ext in supported_image_types]):
        is_image = True
        image_input = cv2.imread(sys.argv[1])
    else:
        image_input = cv2.VideoCapture(sys.argv[1]) #Using the video stream
else:
    print('Usage: python facialrecognition.py [input-video-or-image]')
    exit(1)

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
# gpu_options.allow_growth=False

CHECKPOINT = 100
graph = load_graph("retrained_graph.pb")
graph.finalize()
sess = tf.Session(graph=graph,
    config=tf.ConfigProto(gpu_options=gpu_options))

i = 0
while True:
    if is_image:
        im = image_input
    else:
        (rval, im) = image_input.read()
    try:
        a = im.shape[1]
    except:
        print('Camera not available or input file %s not found or image type not supported! Supported types are %s.'%(sys.argv[1],','.join(supported_image_types) ))
        exit(1)
    if not is_image or (is_image and i == 0):
        # Improve the performance by resizing the image
        mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

        # Detects all the faces
        faces = classifier.detectMultiScale(mini)
        print('Detected %d faces'%len(faces))
        # Draws green rectangles around each detected face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] 
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
            
            #Extracts only the detected face from the image
            zoomed_Face = im[y:y+h, x:x+w]

            CurrentFaceFile = "test.jpg" 
            cv2.imwrite(CurrentFaceFile, zoomed_Face)
            
            # start = datetime.datetime.now()
            text = main(CurrentFaceFile,graph,sess)# Gets the corresponding label (angry,happy,etc.)
            # end=datetime.datetime.now()-start
            # print('Time elapsed: %dms'%(end.microseconds/1000))
            text = text.title()
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(im, text,(x+w,y), font, 1, (0,255,0), 2)
    i += 1
    # Show the image
    cv2.imwrite('output.jpg', im)
    cv2.imshow('Capture',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
