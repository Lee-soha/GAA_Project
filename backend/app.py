from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from joblib import load
import numpy as np
import pandas as pd
import re
import os
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from keras.layers import Input
import openai
import pickle


app = Flask(__name__)
CORS(app)

toon = pd.concat([pd.read_csv('naver.csv'), pd.read_csv('naver_challenge.csv')], ignore_index=True)
results = {}
okt = Okt()
stop_words = set(['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한','그','그녀','속','시작','를','웹','웹툰','툰','웹툰판','이번','부','판','뿐','남자','여자','나','이야기','데','전','후','그들','사람','신작','자신','소년','소녀','만화'])

def preprocessing(text, okt, remove_stopwords = False, stop_words = []):
    ko_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", text)
    word_token = okt.nouns(ko_text)
    word_token = [token for token in word_token if not token in stop_words]
    return ' '.join(word_token) 

# Load the TF-IDF vectorizer and preprocessed data
tfidf_desc_vect = load('tfidf_desc_vect.joblib')
with open('clean_desc_list.pickle', 'rb') as file:
    clean_desc_list = pickle.load(file)
with open('clean_title_list.pickle', 'rb') as file:
    clean_title_list = pickle.load(file)

# 토큰화 파일 불러오기
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# 선학습된 모델 불러오기
model = load_model('best-CNN-model.h5')

@app.route('/get_webtoon_info', methods=['POST'])
def get_webtoon_info():
    data = request.get_json()
    new_title = data['new_title']
    new_description = data['new_description']
    genres = data['genres']
    print(genres)

    # 추천 줄거리 제공
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"I'm planning to create a new webtoon with the title {new_title} and the description {new_description}. Can you write a plot for this story? I don't need a response, just show the plot. Answer in Korean"}]
    )
    plot = response.choices[0].message.content
    results['plot'] = plot

    # 평점 예측
    def predict_rating(title, description, genres):
        combined_sequence = tokenizer.texts_to_sequences([title + ' ' + description])
        padded_combined = pad_sequences(combined_sequence, maxlen=61)
        predicted_rating = model.predict([padded_combined, np.array(genres).reshape(1, -1)])
        return round(float(predicted_rating[0][0]), 2)
    # 유사 웹툰 추천
    def get_recommendations_for_description(description):
        pre_description = preprocessing(description, okt, remove_stopwords=True, stop_words=stop_words)
        overview_tfidf = tfidf_desc_vect.transform([pre_description])
        sim_scores = sorted(list(enumerate(cosine_similarity(overview_tfidf, tfidf_desc_vect.transform(clean_desc_list)).flatten())), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        toon_indices = [i[0] for i in sim_scores]
        recommended_webtoons = toon.loc[toon_indices, 'title'].tolist()
        return recommended_webtoons


    predicted_rating = predict_rating(new_title, new_description, genres)
    recommended_webtoons = get_recommendations_for_description(new_description)
    results['other_info'] = {'predicted_rating': predicted_rating, 'recommended_webtoons': recommended_webtoons}

    # 결과 전송
    print(results)
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
