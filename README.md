# Installing dependencies
To install dependencies run the following command:

``
pip3 install -r requirements.txt
``

# How to retrain the AI
To train the AI, you must perform the following steps.

1. Do an angry face and look at your webcamera
2. Run the command: ``
python3 utils.py train_emotion_from_webcam andre Angry
``

Repeat this process for the following emotions: Angry, Happy, Sad, Calm.

3. Run ``
python3 retrain.py --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --architecture=MobileNet_1.0_224 --image_dir=images
``

# How to run
To run the project using your webcam run:

``
python3 facialrecognition.py
``

Or specify the path of a video or image:

``
python3 facialrecognition.py [video-or-image-path]
``