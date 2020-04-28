# Speech Recognition System Dummy Deployment

Deployment on a local machine(dummy) of a Convolutional Neural Network aimed at the recognition of voice commands.

The architecture bear on Docker linking diffent modules as follows:

Docker[Docker(Flask(Tensorflow) <---> uWSGI)<---->Docker(NGINX)]<----> Client.py

## Install
python packages:
run from terminal---> $pip install -r ./requirements.txt

## Usage

Provided that Docker, uWSGI and NGINX are installed on the server machine (in this case the local one as for the client, hence the "dummy" attribute) simply run:

$python3 Client.py /path-to-a-wav-file.wav

Trained Spot Keywords are: "left","go","up","off","right","down","stop","on","no","yes"

The net was trained and tested on data downloaded from:
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
