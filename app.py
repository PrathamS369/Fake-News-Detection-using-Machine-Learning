import re
import nltk
from flask import Flask, request, render_template
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Download NLTK data ────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',    quiet=True)

# ─── Initialize preprocessing tools ──────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text or ""
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    return ' '.join(
        lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words
    )

# ─── Load model artifacts ─────────────────────────────────────────────────────
vectorizer = joblib.load('tfidf_vectorizer.joblib')
clf        = joblib.load('logreg_fake_news.joblib')

# ─── Flask app setup ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news = request.form.get('news_text')
        clean = clean_text(news)
        vec   = vectorizer.transform([clean])
        pred  = clf.predict(vec)[0]
        label = 'Real' if pred == 1 else 'Fake'
        return render_template('result.html', news=news, prediction=label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
