import re
import json
import time
import tensorflow.keras as kf
import numpy as np
import pickle
from .serializers import TimerSerializer
from .models import Timer, Profile
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from konlpy.tag import Okt
#from konlpy.tag import Mecab
from transformers import TextClassificationPipeline
from transformers import BertTokenizerFast
from transformers import TFBertForSequenceClassification
import pydub
import os
import subprocess

BASE_PATH = "/home/hadoop/rest/model/"

with open(BASE_PATH +'LSTM_tokenizer.pickle', 'rb') as handle:
    LSTMtokenizer = pickle.load(handle)

with open(BASE_PATH + "LSTM.json","r") as f:
    LSTM_INFO = json.load(f)

with open(BASE_PATH +'RNN.pickle', 'rb') as handle:
    Train_list = pickle.load(handle)    
            
LSTMmax_len = LSTM_INFO["maxlen"]
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
LSTM_MODEL = kf.models.load_model(BASE_PATH + 'LSTM_model.h5')
RNN_model = kf.models.load_model(BASE_PATH +'RNN_model.h5') 

def get_okt(text:str):
    tokenized_sentence = okt.morphs(text, stem=False)
    #topwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    return tokenized_sentence
def get_token(okt:list):
    voca_size = 47831
    tokenizer = Tokenizer(voca_size) 
    tokenizer.fit_on_texts(okt)
    X_token = tokenizer.texts_to_sequences(okt)
    return X_token
def get_pad_sequences(X_token:list, max_len):
    return pad_sequences(X_token, maxlen=max_len)
def rnn_result(text:str):
    Train_list.append(get_okt(text))
    Train_token = get_token(Train_list)
    Train_token = pad_sequences(Train_token, 40)
    data = Train_token[-1]
    data = np.array(data).reshape((1, data.shape[0], 1)).astype('float32')
    Train_list.pop()
    return data

def audio_to_text(filename):
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
    text = r.recognize_google(audio_data, language='ko-KR')
    return text
    

loaded_tokenizer = BertTokenizerFast.from_pretrained('/home/hadoop/rest/model/bert_base')
loaded_model = TFBertForSequenceClassification.from_pretrained('/home/hadoop/rest/model/bert_base')

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model, 
    framework='tf',
    return_all_scores=True
)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


class ReflectAPI(APIView):
    def post(self, request):
        if request.method == 'POST':


            print("IP : ",get_client_ip(request))

            cussword = json.loads(request.body)
            
            print("requested : ", cussword["EXAMPLE"])

            user_input = cussword["EXAMPLE"].replace("\n",".").split(sep = '.')
            tosend = user_input
            RESULT = {}

            for i in range(len(user_input)):
                user_input[i] = re.sub(r"^[0-9]+,", "", user_input[i])
                user_input[i] = user_input[i].replace("dc App","")
                user_input[i] = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', user_input[i])
                user_input[i] = re.sub('[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s]', ' ', user_input[i])
                user_input[i] = re.sub(r"ㅋ+", " ", user_input[i])
                user_input[i] = re.sub(r"\s+", " ", user_input[i]).strip()
                    
            

                #@@@@@@@@@@@@@@@@@@LSTM@@@@@@@@@@@@@@@@@@@@


                LSTMtokenize = okt.morphs(user_input[i])
                LSTMtokenize = [word for word in LSTMtokenize if not word in stopwords]
                LSTMtokenize = LSTMtokenizer.texts_to_sequences([LSTMtokenize])
                LSTMtokenize = pad_sequences(LSTMtokenize, maxlen=LSTMmax_len)


                LSTM_START = time.time()
                LSTM_PRO = LSTM_MODEL.predict(LSTMtokenize)
                LSTM_END = time.time()
                LSTM_PRO = float(LSTM_PRO)
                
                LSTM_TIME = LSTM_END - LSTM_START


                #@@@@@@@@@@@@@@@@@@@RNN@@@@@@@@@@@@@@@@@@@@
                rnn_submit = rnn_result(user_input[i])
                RNNstart = time.time()
                RNNAccuracy = RNN_model.predict(rnn_submit)
                RNNstop = time.time()

                RNNrunningtime = RNNstop - RNNstart

                #@@@@@@@@@@@@@@@@@@@BERT@@@@@@@@@@@@@@@@@@@@

                BERTstart = time.time()
                BERTAccuracy = text_classifier(user_input[i])
                BERTstop = time.time()
                BERTrunningtime = BERTstop - BERTstart
                BERTAccuracy = BERTAccuracy[0]

                BERTupdateAccNTime = {
                    "probability":BERTAccuracy[1]['score'],
                    "time":BERTrunningtime
                }

                SISI = {
                    "EXAMPLE":tosend[i],
                    "LSTM-probability":LSTM_PRO,
                    "LSTM-time":LSTM_TIME,
                    "RNN-probability":RNNAccuracy[0][1],
                    "RNN-time":RNNrunningtime,
                    "BERT-probability":BERTAccuracy[1]['score'],
                    "BERT-time":BERTrunningtime
                }

                RESULT[i] = (SISI)
                
                print({'result':RESULT})
        
            return Response({'result':RESULT}) 


class AudioAPI(APIView):
    def post(self, request):
        print(request.body)
        r = json.loads(request.body)

        try:
            with open("test.mp4" ,"wb" ) as f:
                f.write(bytearray.fromhex(r['file']))


            command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn -y {}".format("test.mp4",  "convert.wav")
            subprocess.call(command, shell=True)
            converted = audio_to_text('convert.wav')

            print("IP : ",get_client_ip(request))

            user_input[i] = converted

            print("requested : ", converted)

            user_input[i] = re.sub(r"^[0-9]+,", "", user_input[i])
            user_input[i] = user_input[i].replace("dc App","")
            user_input[i] = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', user_input[i])
            user_input[i] = re.sub('[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s]', ' ', user_input[i])
            user_input[i] = re.sub(r"ㅋ+", " ", user_input[i])
            user_input[i] = re.sub(r"\s+", " ", user_input[i]).strip()
                
        

            #@@@@@@@@@@@@@@@@@@LSTM@@@@@@@@@@@@@@@@@@@@


            LSTMtokenize = okt.morphs(user_input[i])
            LSTMtokenize = [word for word in LSTMtokenize if not word in stopwords]
            LSTMtokenize = LSTMtokenizer.texts_to_sequences([LSTMtokenize])
            LSTMtokenize = pad_sequences(LSTMtokenize, maxlen=LSTMmax_len)


            LSTM_START = time.time()
            LSTM_PRO = LSTM_MODEL.predict(LSTMtokenize)
            LSTM_END = time.time()
            LSTM_PRO = float(LSTM_PRO)
            
            LSTM_TIME = LSTM_END - LSTM_START


            #@@@@@@@@@@@@@@@@@@@RNN@@@@@@@@@@@@@@@@@@@@
            rnn_submit = rnn_result(user_input[i])
            RNNstart = time.time()
            RNNAccuracy = RNN_model.predict(rnn_submit)
            RNNstop = time.time()

            RNNrunningtime = RNNstop - RNNstart

            #@@@@@@@@@@@@@@@@@@@BERT@@@@@@@@@@@@@@@@@@@@

            BERTstart = time.time()
            BERTAccuracy = text_classifier(user_input[i])
            BERTstop = time.time()
            BERTrunningtime = BERTstop - BERTstart
            BERTAccuracy = BERTAccuracy[0]

            BERTupdateAccNTime = {
                "probability":BERTAccuracy[1]['score'],
                "time":BERTrunningtime
            }

            RESULT = {
                "EXAMPLE":converted,
                "LSTM-probability":LSTM_PRO,
                "LSTM-time":LSTM_TIME,
                "RNN-probability":RNNAccuracy[0][1],
                "RNN-time":RNNrunningtime,
                "BERT-probability":BERTAccuracy[1]['score'],
                "BERT-time":BERTrunningtime
            }
            print(RESULT)
            return Response(RESULT) 
        except:
            RESULT = {
                "ERROR":True
            }
            return Response(RESULT) 

        