import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import re

# 불용어 로드
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords

stopwords = load_stopwords('stopword.txt')

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)  # 줄바꿈 제거
    text = re.sub(r'[^가-힣\s]', '', text)  # 한글 및 공백 제외 제거
    text = text.lower()  # 소문자 변환 (한국어는 필요 없지만 일관성 유지)

    # 형태소 분석기 초기화
    okt = Okt()
    tokens = okt.morphs(text, stem=True)  # 형태소 분석 및 어간 추출
    tokens = [word for word in tokens if word not in stopwords]  # 불용어 제거
    return ' '.join(tokens)

# UI 구성
st.set_page_config(page_title="News Keywords", page_icon=":newspaper:")

# Custom CSS for styling
st.markdown("""
    <style>
        .header {
            text-align: center;
            background: linear-gradient(to right, #d4fc79, #96e6a1);
            -webkit-background-clip: text;
            color: transparent;
            font-size: 4em;
            font-weight: bold;
            margin-bottom: 20px;  /* Reduced margin */
        }
        .main-description {
            text-align: center;
            margin-bottom: 50px;
        }
        .search-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }
        .search-input {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 20px 0 0 20px;
            outline: none;
            width: 300px;
        }
        .search-button {
            padding: 10px;
            border: none;
            border-radius: 0 20px 20px 0;
            background: #1e90ff;
            color: white;
            cursor: pointer;
            outline: none;
        }
    </style>
""", unsafe_allow_html=True)

# Render the header
st.markdown("<div class='header'>NEWS KEYWORDS</div>", unsafe_allow_html=True)

# Render the main description
st.markdown("""
    <div class='main-description'>
        <h3>시사 키워드 리스트 서비스</h3>
        <p>키워드로 알아보는 최신 핵심 뉴스</p>
    </div>
""", unsafe_allow_html=True)


# Render the category buttons in a single row without wrapping
st.markdown("<div class='categories'>", unsafe_allow_html=True)
categories = ["검색하기", "경제", "정치", "생활/문화", "IT/과학", "세계", "스포츠", "연예"]
cols = st.columns(len(categories))

for col, category in zip(cols, categories):
    with col:
        if st.button(category, key=category):
            st.write(f"Selected category: {category}")
st.markdown("</div>", unsafe_allow_html=True)

# Input field for article content
article_content = st.text_area("기사 내용을 입력하세요")


# 불용어 로드
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords

stopwords = load_stopwords('stopword.txt')



# 형태소 분석기 초기화
okt = Okt()

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)  # 줄바꿈 제거
    text = re.sub(r'[^가-힣\s]', '', text)  # 한글 및 공백 제외 제거
    text = text.lower()  # 소문자 변환 (한국어는 필요 없지만 일관성 유지)

    tokens = okt.morphs(text, stem=True)  # 형태소 분석 및 어간 추출
    tokens = [word for word in tokens if word not in stopwords]  # 불용어 제거
    return ' '.join(tokens)





# Button to trigger keyword extraction
if st.button("검색"):
    # 전처리
    preprocessed_text = preprocess_text(article_content)

    # TF-IDF 벡터화
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([preprocessed_text])

    # TF-IDF 결과를 데이터프레임으로 변환
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # 상위 n개 키워드 추출
    n = 5  # 상위 5개 키워드 추출
    top_keywords = tfidf_df.iloc[0].sort_values(ascending=False).head(n).index.tolist()

    # 결과 출력
    st.subheader("추출된 키워드:")
    st.write(", ".join(top_keywords))
