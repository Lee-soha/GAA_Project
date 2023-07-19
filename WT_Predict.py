import pandas as pd
import re
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from keras.layers import Input
import openai
import pickle
from joblib import load

toon = pd.concat([pd.read_csv('naver.csv'), pd.read_csv('naver_challenge.csv')], ignore_index=True)
def get_webtoon_info(new_title, new_description):
    # 함수 전처리
    okt = Okt()
    stop_words = set(['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한','그','그녀','속','시작','를','웹','웹툰','툰','웹툰판','이번','부','판','뿐','남자','여자','나','이야기','데','전','후','그들','사람','신작','자신','소년','소녀','만화'])
    
    def preprocessing(text, okt, remove_stopwords = False, stop_words = []):
        desc_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", text)
        word_desc = okt.nouns(desc_text)
        clean_desc = [token for token in word_desc if not token in stop_words]
        return ' '.join(clean_desc) 

    # Load the TF-IDF vectorizer and preprocessed data
    tfidf_desc_vect = load('tfidf_desc_vect.joblib')
    with open('clean_desc_list.pickle', 'rb') as file:
        clean_desc_list = pickle.load(file)
    with open('clean_title_list.pickle', 'rb') as file:
        clean_title_list = pickle.load(file)

    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)

    # Load the pre-trained model
    model = load_model('best-LSTM-model.h5')

    # 평점 예측 함수
    def predict_rating(title, description):
        title_sequence = tokenizer.texts_to_sequences([title])
        padded_title = pad_sequences(title_sequence, maxlen=19)
        description_sequence = tokenizer.texts_to_sequences([description])
        padded_description = pad_sequences(description_sequence, maxlen=250)
        predicted_rating = model.predict([padded_title, padded_description])
        return predicted_rating[0][0]

    # 유사 웹툰 함수
    def get_recommendations_for_description(description):
        pre_description = preprocessing(description, okt, remove_stopwords=True, stop_words=stop_words)
        overview_tfidf = tfidf_desc_vect.transform([pre_description])
        sim_scores = sorted(list(enumerate(cosine_similarity(overview_tfidf, tfidf_desc_vect.transform(clean_desc_list)).flatten())), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        toon_indices = [i[0] for i in sim_scores]
        recommended_webtoons = toon.loc[toon_indices, 'title'].tolist()
        return recommended_webtoons

    # 추천 줄거리 제공
    openai.api_key = "OPENAI_API_KEY"
    response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[{"role": "user", "content": f"I'm planning to create a new webtoon with the title {new_title} and the description {new_description}. Can you write a plot for this story? I don't need a response, just show the plot. Answer in Korean"}]
    )
    generated_plot = response.choices[0].message.content

    # 출력
    predicted_rating = predict_rating(new_title, new_description)
    recommended_webtoons = get_recommendations_for_description(new_description)
    
    print(f"'{new_title}'로 부터 예측된 평점은 : {predicted_rating} 점 입니다.")
    print("\n유사한 웹툰은 :")
    for webtoon in recommended_webtoons:
        print(webtoon)
    print("\n이 웹툰에 대한 추천 플롯은:")
    print(generated_plot)

# 함수 테스트
new_title = "이세계에서 다시한번"
new_description = "어느날 트럭에 치여 이세계로 떨어진 고교생이 생존을 위해 고군분투하는 이야기"
get_webtoon_info(new_title, new_description)
