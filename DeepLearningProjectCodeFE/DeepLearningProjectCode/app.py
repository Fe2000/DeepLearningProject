from flask import Flask, render_template, url_for, request, redirect , send_file
import tensorflow as tf
from tensorflow import keras
import pickle
from keras.preprocessing.sequence import pad_sequences
import PIL
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model



UPLOAD_FOLDER = 'env/static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif' , 'png'}

app = Flask(__name__,)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['POST','GET'])
def index():
    
    modelFrank = keras.models.load_model('inceptionFE.h5')

    def preprocess(image_path):
        # Convert all the images to size 299x299 as expected by the inception v3 model
        img = keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
        # Convert PIL image to numpy array of 3-dimensions
        x = keras.preprocessing.image.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # pre-process the images using preprocess_input() from inception module
        x = keras.applications.inception_v3.preprocess_input(x)
        return x

    def encode(image):
        image = preprocess(image) # preprocess the image
        fea_vec = modelFrank.predict(image) # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
        return fea_vec

    if request.method == 'POST':

        file = request.files['file']



        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        fPath = 'env/static/'  + filename

        image = encode(fPath)

        modelFE =  keras.models.load_model('model_FE9.h5')
        max_length = 33
        words_to_index = pickle.load(open("words.pkl", "rb"))
        index_to_words = pickle.load(open("words1.pkl", "rb"))       

        imageFE = image.reshape(1,2048)


        def Image_Caption(picture):
            in_text = 'startseq'
            for i in range(max_length):
                sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
                sequence = pad_sequences([sequence], maxlen=max_length)
                yhat = modelFE.predict([picture,sequence], verbose=0)
                yhat = np.argmax(yhat)
                word = index_to_words[yhat]
                in_text += ' ' + word
                if word == 'endseq':
                    break
            final = in_text.split()
            final = final[1:-1]
            final = ' '.join(final)
            return final

        predictFE = Image_Caption(imageFE)

        values = [ 
            {
                "pre": predictFE,
                "path": filename
            }
            ]


        return render_template('output.html' , values=values)


    t = "tom added new lines"

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
