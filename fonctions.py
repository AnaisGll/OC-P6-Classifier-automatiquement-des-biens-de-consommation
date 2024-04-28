import pandas as pd
from pandas.api.types import CategoricalDtype
from pandasql import sqldf
import numpy as np
import fonctions as fc
import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import time

from scipy import stats 
from scipy.stats import f_oneway, kruskal
from scipy.stats import shapiro, kstest, yeojohnson, boxcox

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, adjusted_rand_score

from glob import glob
import keras.preprocessing.image

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
import gensim
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel, TFAutoModel, AutoTokenizer

import re
import string
import os
from os import listdir

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, words, wordnet, brown
from nltk.tag import pos_tag

from wordcloud import WordCloud
from PIL import Image
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

import cv2
from PIL import Image, ImageOps, ImageFilter
from IPython.display import Image, display


''' Fonctions de prétraitement de texte'''

def extract_unique_words(text):
    '''Extraction des mots uniques'''
    words = set(text.lower().split())
    return words




def display_tokens_info(tokens):
    
    '''Renseignement sur le nombre de tokens dans le corpus'''
    
    print(f'nb tokens : {len(tokens)}, nb tokens uniques : {len(set(tokens))}')
    print(tokens[:50])




def process_description(doc, rejoin=False):
    
    '''
    Fonction basique de nettoyage de texte
    '''

    # Mise en minuscule
    doc = doc.lower().strip()

    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')  
    tokens = tokenizer.tokenize(doc)

    # Élimination des stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

    if rejoin:
        return " ".join(lemmatized_tokens)
    
    return lemmatized_tokens




def pre_cleaning(doc):
    
    '''Pré-nettoyage pour la description de chaque produit'''
    
    new_variable = process_description(doc, rejoin=True)
    
    return new_variable




import re

def clean_text(text):
    '''
    Nettoyage du texte en éliminant les éléments des adresses URL, les balises HTML et les caractères non-ASCII.
    '''
    
    # Élimination des éléments des adresses URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Élimination des éléments HTML du texte
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    text = re.sub(html, "", text)
    
    # Élimination des caractères non-ASCII
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    
    return text


''' Fonctions BERT'''

def bert_inp_fct(sentences, bert_tokenizer, max_length):
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens=True,
                                              max_length=max_length,
                                              padding='max_length',
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")

        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0],
                             bert_inp['token_type_ids'][0],
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)

    return input_ids, token_type_ids, attention_mask, bert_inp_tot

def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF'):
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx+batch_size],
                                                                              bert_tokenizer, max_length)

        if mode == 'HF':    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode == 'TFhub': # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids,
                                 "input_mask" : attention_mask,
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']

        if step == 0:
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else:
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot, last_hidden_states))

    features_bert = np.array(last_hidden_states_tot).mean(axis=1)

    time2 = np.round(time.time() - time1, 0)
    print("temps traitement : ", time2)

    return features_bert, last_hidden_states_tot

''' METHODE USE'''

def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
    return features


'''Fonction de traitement des images'''

def image_pathway(image):
    
    '''Fonction permettant le chargement des images'''
    
    image = "Images/" + image
    
    return image



def grayscale_images(image):

    '''Preprocessing des images avec conversion en nuance de gris'''
    file_dir = os.path.split(image)
    
    # Chargement de l'image avec conversion en nuance de gris
    img = Image.open(image).convert('L')
    
    # Réglage du contraste
    img = ImageOps.equalize(img)
    
    # Application du filtre médian
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Création du répertoire
    output_dir = "Images_grayscale"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, file_dir[1])
    img.save(output_path)
    
    return output_path


def processed_images(image):
    
    '''Preprocessing des images sans conversion en nuance de gris'''

    file_dir = os.path.split(image)
    
    # Chargement de l'image sans conversion en nuance de gris
    img = Image.open(image)
    
    # Réglage du contraste
    img = ImageOps.equalize(img)
    
    # Application du filtre médian
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Création du répertoire
    output_dir = "Images_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, file_dir[1])
    img.save(output_path)
    
    return output_path

def resize_image(image_path, target_size=(224, 224)):
    """Redimension des images en 224x224"""
    image = Image.open(image_path)
    resized_image = image.resize(target_size, Image.NEAREST)
    return resized_image

from PIL import Image

def processed_images(image):
    
    '''Preprocessing des images sans conversion en nuance de gris'''

    file_dir = os.path.split(image)
    
    # Chargement de l'image sans conversion en nuance de gris
    img = Image.open(image)
    
    # Réglage du contraste
    img = ImageOps.equalize(img)
    
    # Application du filtre médian
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Création du répertoire
    output_dir = "Images_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, file_dir[1])
    img.save(output_path)
    
    return output_path