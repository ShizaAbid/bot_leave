#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import LSTM 
from keras.layers import Embedding, Dense, Dropout


# # Function to perform pre-processing steps

# At first tokenization will be performed then stemming will be performed over tokens

# In[4]:


def token_stems(text):
        
    tokens=tokenizing(text) 
    stems=stemming(tokens)
    return stems       


# # Stemming Function

# In[5]:


from nltk.stem.snowball import SnowballStemmer
stemmer= SnowballStemmer("english")


def stemming(text):
    
    stems =[stemmer.stem(t) for t in text]
    return stems


# # Tokenization Function

# In[6]:


def tokenizing(text):
    
    #breaking each word and making them tokens
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    #storing only alpha tokens
    filtered_tokens=[]
    for token in tokens:
        if (re.search('[a-zA-Z]|\'', token)):
            filtered_tokens.append(token)

    return filtered_tokens


# In[7]:


train= pd.read_excel("C:\\Users\\shiza.abid\\Desktop\\DataSetLeave_FinalMerge.xlsx")
#train.shape


# In[8]:


docs= train['Leave Data Description']


# In[9]:


tokens = []
for i in docs:
    temp = token_stems(i)
    tokens.append(temp)


# In[10]:


#print(docs[1],"\n\n",tokens[1])


# In[11]:


x, y = np.asarray(tokens) , np.asarray(train['Class'])


# In[12]:


le = LabelEncoder()
y = le.fit_transform(y)
y= to_categorical(y)
#y[:10]


# In[13]:


xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)


# In[14]:


max_words = 20000
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x)


# In[15]:


sequences = tok.texts_to_sequences(x)
test_sequences = tok.texts_to_sequences(xtest)


# In[16]:


max_len =200

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#test_sequences_matrix[1]


# # Model 

# In[17]:


def model():
    
    #Create model by simply calling the Sequential constructor
    
    model = Sequential()
    
    #This layer can only be used as the first layer in a model.
    
    model.add(Embedding(max_words,50,input_length=max_len))
    
    #specify number of neurons in lstm layer and the activation function that we want to use.
    
    model.add(LSTM(64, activation='tanh',dropout=0.01))
    
    #specify number of neurons in dense layer and the activation function that we want to use.
    
    model.add(Dense(50, activation='relu'))
    
    #add dropout to prevent overfitting
    
    model.add(Dropout(0.01))
    
    #and we create our output layer with two nodes as we have 2 class labels.
    
    print(model.add(Dense(2, activation='softmax')))
    
    #Now for training, we need to define an optimizer, loss measure and the error metric.
    # We will use the binary_crossentropy as our loss measure. 
    #As for the minimization algorithm, we will use "rmsprop".
    #This optimizer is usually a good choice for recurrent neural networks.
    
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    return model


# In[26]:


m= model()

m.fit(sequences_matrix,y,epochs=12, validation_split=0.2)


# In[27]:


accuracy = m.evaluate(test_sequences_matrix,ytest)


# In[28]:


#save your model
m.save('Leave_or_inquiry_model.h5')


# In[ ]:





# In[ ]:




