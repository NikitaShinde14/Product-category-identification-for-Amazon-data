#!/usr/bin/env python
# coding: utf-8
# to use if zip file was uploaded

import gzip
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# librariesto be saved in requirements.txt format

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
import re
import string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize,sent_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential

beauty_df = pd.read_json('/content/drive/MyDrive/nlp_amazon_proj/Beauty_5.json',lines=True)

from google.colab import drive
drive.mount('/content/drive')
beauty_df.head()

#cellphone_df = getDF('Cell_Phones_and_Accessories_5.json.gz')
cellphone_df = pd.read_json('/content/drive/MyDrive/nlp_amazon_proj/Cell_Phones_and_Accessories_5.json',lines=True)
cellphone_df.head()

grocery_df = pd.read_json('/content/drive/MyDrive/nlp_amazon_proj/Grocery_and_Gourmet_Food_5.json',lines=True)
grocery_df.head()

homeKitchen_df = pd.read_json('/content/drive/MyDrive/nlp_amazon_proj/Home_and_Kitchen_5.json',lines=True)
homeKitchen_df

sports_df = pd.read_json('/content/drive/MyDrive/nlp_amazon_proj/Sports_and_Outdoors_5.json',lines=True)
sports_df

beauty_df = beauty_df[['reviewText','summary']]
beauty_df['Category'] = 'Beauty'
beauty_df


cellphone_df = cellphone_df[['reviewText','summary']]
cellphone_df['Category'] = 'Cellphones'
cellphone_df

grocery_df = grocery_df[['reviewText','summary']]
grocery_df['Category'] = 'Grocery_GourmetFood'
grocery_df

homeKitchen_df = homeKitchen_df[['reviewText','summary']]
homeKitchen_df['Category'] = 'home_kitchen'
homeKitchen_df

sports_df = sports_df[['reviewText','summary']]
sports_df['Category'] = 'sports_outdoors'
sports_df

df = pd.concat([beauty_df,cellphone_df,grocery_df,homeKitchen_df,sports_df])
df

df['Category'].value_counts().plot(kind = 'barh')
df['Category'].value_counts().min()

df.drop_duplicates(inplace = True)

df['Category'].value_counts().min()  # now we can take 150000 records to have a balance for entire data
df = df.groupby('Category').head(150000)

df.shape
df.head()

df['reviewText']  = df[['reviewText','summary']].apply(lambda x : ' '.join(x),axis = 1)
df['reviewText'].iloc[0]

df.drop('summary',axis = 1, inplace = True)
df

# cleaning the data
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]',' ',text)
    text = re.sub('http\S+',' ',text,flags=re.MULTILINE)   # to clean the http and https links,for multiple sentences (if review has many sentences) as well.
    text = re.sub('\.+',' ',text)                      # remove ..... if any
    word_tokens = word_tokenize(text)
    text = [words for words in word_tokens if words not in stops]   # clean text
    text = [words for words in text if words not in string.punctuation]  # to remove any punctuation marks
    return ' '.join(text)

try_text = 'i got this mobile....and it is working for 5 days!!! happy till today :D'

clean_text(try_text)

df['cleaned_review'] = df['reviewText'].apply(lambda x: clean_text(x))
df['cleaned_review']

df.head()

df.drop('reviewText',axis=1,inplace=True)

Category = {'Beauty':0,'sports_outdoors':1,'Cellphones':2,'home_kitchen':3,'Grocery_GourmetFood':4}
df['Category'] = df['Category'].map(Category)
df

train_beauty = df[df['Category']==0][:120000] #selectin 120000 entries from each category
train_sports = df[df['Category']==1][:120000]

train_cellphone = df[df['Category']==2][:120000]
train_home = df[df['Category']==3][:120000]
train_grocery = df[df['Category']==4][:12000]
test_beauty = df[df['Category']==0][120000:]
test_sports = df[df['Category']==1][120000:]
test_cellphone = df[df['Category']==1][120000:]
test_home = df[df['Category']==1][120000:]
test_grocery = df[df['Category']==1][120000:]

train_df = pd.concat([train_beauty,train_sports,train_cellphone,train_home,train_grocery])

test_df = pd.concat([test_beauty,test_sports,test_cellphone,test_home,test_grocery])

train_df.head()

test_df.head()

train = train_df.sample(frac=1.0) # suffling the train 

train['cleaned_review'][0]
test = test_df.sample(frac=1.0)

x_train = train['cleaned_review']
x_test = test['cleaned_review']
y_train = train['Category']
y_test = test['Category']

x_train.head()
y_train.head()

tokenizer = Tokenizer(oov_token='<unk>')
tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index
len(word_index)

vocab_size = len(word_index) +1
vocab_size

plt.hist(df['cleaned_review'].apply(lambda x : len(x.split())),range=(0,200))
plt.show()

# padding
max_length = 80
embedding_dimension = 300
truncating_type = 'post'
padding_type = 'post'

train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

word_index['red']

print(train_sequences[0])
print(x_train[0])

len(train_sequences[0])
len(x_train[0])

train_padded = pad_sequences(train_sequences,maxlen=max_length,padding=padding_type,truncating=truncating_type)
test_padded = pad_sequences(test_sequences,maxlen=max_length,padding=padding_type,truncating=truncating_type)

train_padded[0]
len(train_padded[0])

rev_word_index = dict([(value,key) for (key,value) in word_index.items()])
rev_word_index[273]

def decode_sentence(number):
    return ' '.join([rev_word_index.get(i,'?') for i in number])
decode_sentence(train_padded[0])

#Model building
lstm_model = Sequential([

    tf.keras.layers.Embedding(vocab_size,embedding_dimension),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(100,activation = 'relu'),
    tf.keras.layers.Dense(5,activation = 'softmax')
        
])

lstm_model.summary()
opt = tf.keras.optimizers.Adam(0.01)   
lstm_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics= ['accuracy'])
num_epochs = 3
history = lstm_model.fit(train_padded,y_train,epochs=3,validation_data=(test_padded,y_test),batch_size = 1000)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('Accuracy Curve')
plt.legend(['Training','Testing'],loc="best")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title('loss Curve')
plt.legend(['Training','Testing'],loc="best")

lstm_model.save('LSTM_Model.h5')

import pickle

with open('tokenizer.pickle','wb') as handle:   #save it in a notepad
  pickle.dump(tokenizer,handle,protocol = pickle.HIGHEST_PROTOCOL)

with open('tokenizer.pickle','rb') as pr:   #save it in a notepad
    load_tokenizer = pickle.load(pr)

test = ['asian hair medium thickness hair uncurls get side head used ones better lasted one day without sprays one might work fine hair work']

test_to_sequences = load_tokenizer.texts_to_sequences(test)

test_to_sequences

test_padded_sequence = pad_sequences(test_to_sequences,maxlen = 80 ,padding = 'post',truncating = 'post')

new_model = tf.keras.models.load_model('./LSTM_Model.h5')

new_model.summary()
pred = new_model.predict(test_padded_sequence)

pred #softmax

#category = {'Beauty':0,'sports':1,'cellphone':2,'home':3,'grocery':4}

labels = list(category.keys())

labels

labels[np.argmax(pred)]