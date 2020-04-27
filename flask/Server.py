# CLIENT <--> uWSGI <--> Flask <--> Tensorflow
###### Terminal
# conda install -c conda-forge uswgi
# conda install -c conda-forge libiconv
# uwsgi --http 127.0.0.1:5050 --wsgi-file /PATH-TO/Server.py --callable app --processes 1 --threads 1 --master True
##### OR simply use the conf file app.ini:
#uwsgi app.ini

from flask import Flask, request,jsonify
import random
from Keyword_Spotting_Service import Keyword_Spotting_Service
import os

app = Flask(__name__)

@app.route("/predict", methods =["POST"])

def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

    # invoke keyword spotting system
    kss = Keyword_Spotting_Service()

    # make a prediction
    predicted_keyword = kss.predict(file_name)

    # remove the audio file from current directory
    os.remove(file_name)

    # send back the predicted keyword in json format
    data ={"keyword":predicted_keyword}

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)