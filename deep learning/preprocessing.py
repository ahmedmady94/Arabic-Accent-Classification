import numpy as np
import joblib
import nltk
import json
import tensorflow as tf

# Loading our model
model = tf.keras.models.load_model('model.h5')


# Loading our word dictionary
with open('worddict.json', 'r') as fp:
    word2idx = json.load(fp)

    
# Loading labelencoder to inverse transform our labels    
le=joblib.load('labelencoder')

# Cleaning data
def clean_data(user_input):
    tokenizer = nltk.RegexpTokenizer(r"[\u0621-\u064A]+")
    clean_words = [tokenizer.tokenize(user_input[i]) for i in range(len(user_input))]
    clean_list=[' '.join(i) for i in clean_words]
    return clean_list

# Making predictions
def predict(text_list, clf, dictionary, padding_size,label_encoder):
    store=[]
    for text in text_list:
        # padd the text
        padded_text = np.zeros((padding_size))
        # transform your text into indices
        padded_text[:min(padding_size, len(text.split()))] = [
            word2idx.get(word, 0) for word in text.split()][:padding_size]
        # predict it !
        prediction = np.argmax(clf.predict(tf.expand_dims(padded_text, 0)),axis=1)
        label= label_encoder.inverse_transform(prediction)
        store.append(label[0])
    return store


# Defining final function
def predict_accent(user_input):
    return predict(user_input,model,word2idx,20,le)