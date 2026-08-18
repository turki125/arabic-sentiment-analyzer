import streamlit as st
import joblib
import re

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_arabic_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title="محلل المشاعر", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .block-container {
        max-width: 1000px;
        margin: 0 auto;
        padding-top: 3rem;
        direction: rtl;
        text-align: right;
    }
    h1 {
        color: #F5F5F7 !important;
        font-weight: 800 !important;
        text-align: right !important;
        direction: rtl !important;
    }
    h2, h3 {
        color: #E5E5E7 !important;
        text-align: right !important;
        direction: rtl !important;
    }
    p, label {
        text-align: right !important;
        direction: rtl !important;
    }
    .subtitle {
        color: #9199A8;
        font-size: 16px;
        line-height: 1.8;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 17px;
        text-align: right;
        direction: rtl;
        background-color: #1A1F2B !important;
        border: 1px solid #2A3040 !important;
        border-radius: 12px !important;
        color: #F5F5F7 !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid #2DD4BF !important;
        box-shadow: 0 0 0 1px #2DD4BF !important;
    }
    .stButton button {
        border-radius: 8px;
        padding: 0.6rem 2.2rem;
        font-weight: 600;
        float: right;
        background-color: #2DD4BF !important;
        color: #0E1117 !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #26B8A5 !important;
    }
    .result-card {
        background-color: #1A1F2B;
        border: 1px solid #2A3040;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-positive {
        color: #2DD4BF;
        font-size: 24px;
        font-weight: 700;
    }
    .result-negative {
        color: #F87171;
        font-size: 24px;
        font-weight: 700;
    }
    .confidence-text {
        color: #9199A8;
        font-size: 15px;
        margin-top: 0.5rem;
    }
    .placeholder-card {
        background-color: #1A1F2B;
        border: 1px dashed #2A3040;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #6B7280;
    }
    .footer {
        text-align: center !important;
        color: #6B7280;
        font-size: 14px;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #2A3040;
        direction: rtl;
    }
    .footer a { color: #2DD4BF !important; }
    hr { border-color: #2A3040 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("محلل المشاعر العربي")
st.markdown(
    '<p class="subtitle">أداة ذكاء اصطناعي تحدد ما إذا كان النص العربي إيجابيًا أو سلبيًا. '
    'تم تدريبها على 8,364 تقييم حقيقي لمطاعم عربية باستخدام TF-IDF والانحدار اللوجستي، '
    'وحققت دقة 82% على بيانات لم يسبق للنموذج رؤيتها.</p>',
    unsafe_allow_html=True
)

st.divider()

col_result, col_input = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("جرّب بنفسك")
    user_input = st.text_area("اكتب جملتك هنا:", height=150, placeholder="اكتب رأيك عن مطعم أو تجربة...")
    analyze_clicked = st.button("تحليل", type="primary")

with col_result:
    st.subheader("النتيجة")
    if analyze_clicked:
        if user_input.strip() == "":
            st.warning("الرجاء إدخال نص.")
        else:
            clean = clean_arabic_text(user_input)
            vec = vectorizer.transform([clean])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = max(prob)

            label = "إيجابي 😊" if pred == 1 else "سلبي 😞"
            css_class = "result-positive" if pred == 1 else "result-negative"

            st.markdown(f"""
                <div class="result-card">
                    <div class="{css_class}">{label}</div>
                    <div class="confidence-text">نسبة الثقة: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.progress(confidence)
    else:
        st.markdown(
            '<div class="placeholder-card">ستظهر النتيجة هنا بعد الضغط على تحليل</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="footer">تم التطوير بواسطة تركي · انحدار لوجستي + TF-IDF · '
    '<a href="https://github.com/turki125/arabic-sentiment-analyzer">عرض على GitHub</a></div>',
    unsafe_allow_html=True
)