# Speech Recognition System Dummy Deployment

Deployment on a local machine(dummy) of a Convolutional Neural Network aimed at the recognition of voice commands.

The architecture bear on Docker linking diffent modules as follows:

<<<<<<< HEAD
Docker[Docker(Flask(Tensorflow) <---> uWSGI)<---->Docker(NGINX)]<----> Client.py

## Install
python packages:
run from terminal---> $pip install -r ./requirements.txt

## Usage

Provided that Docker, uWSGI and NGINX are installed on the server machine (in this case the local one as for the client, hence the "dummy" attribute) simply run:

$python3 Client.py /path-to-a-wav-file.wav

Trained Spot Keywords are: "left","go","up","off","right","down","stop","on","no","yes"

The net was trained and tested on data downloaded from:
https://www.youtube.com/redirect?v=VPJ2jazh_KI&redir_token=JqIv5xkdyEWNEuKbqcfnzMfcoK18MTU4ODE4MzMyNEAxNTg4MDk2OTI0&event=video_description&q=https%3A%2F%2Fai.googleblog.com%2F2017%2F08%2Flaunching-speech-commands-dataset.html

=======
Python Tensorflow <---> Flask <----> uWSGI <-----> NGINX
>>>>>>> 02ef269eb4376286de850498c29c5ca984d06df9

Developped following Valerio Velardo notes (The Sound of AI): https://github.com/musikalkemist/
