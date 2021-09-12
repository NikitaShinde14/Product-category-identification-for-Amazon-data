import pandas as pd
import numpy as np
import tensorflow as tf
from google.protobuf.descriptor import MethodDescriptor
import pickle
import requests
from flask import Flask,request,jsonify,render_template 
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask("Category Model")

global category
global label

category = {'Beauty':0,'sports':1,'cellphone':2,'home':3,'grocery':4}
label = list(category.keys())


max_len = 80
embeding_dimension = 300
trunc_type = "post"
pad_type="post"


new_model = tf.keras.models.load_model('./model/LSTM_Model.h5')



with open('./model/tokenizer.pickle','rb') as pr:   #save it in a notepad
    loaded_tokenizer = pickle.load(pr)


def predict_category(text):
    text_sequence = loaded_tokenizer.texts_to_sequences(text)
    pad_text_sequence = pad_sequences(text_sequence,padding=pad_type,truncating = trunc_type,maxlen=max_len)
    pred_new = new_model.predict_classes(pad_text_sequence)
    return label[np.argmax(pred_new)]

@app.route('/')
def home():
    return render_template('form.html')


@app.route('/result',methods=["POST"])
def result():
    if request.method == "POST":
        text = request.form['input']
        predicted_category = predict_category(text)
        return render_template('result.html',text=text,predicted_category=predicted_category)



if __name__ == '__main__':
    app.run(host="localhost",port=5000,debug=True)
    


