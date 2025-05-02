# Fake News Detection Flask App

A simple web application that classifies news articles as **Real** or **Fake** using a Machine Learning pipeline (TF‑IDF + Logistic Regression) and serves predictions via a Flask interface.

---

## 🧰 Project Structure

```
fake_news_app/
├── app.py
├── tfidf_vectorizer.joblib
├── logreg_fake_news.joblib
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
├── images/
│   ├── screenshot_index.png
│   └── screenshot_result.png
└── requirements.txt
```

* **app.py**: Main Flask application.
* **.joblib** files: Serialized TF‑IDF vectorizer and trained Logistic Regression model.
* **static/style.css**: Custom CSS for styling.
* **templates/**: HTML templates for input form and result page.
* **images/**: Screenshots demonstrating the UI.
* **requirements.txt**: Lists all Python dependencies.

---

## ⚙️ Dependencies

All dependencies are pinned in `requirements.txt`. Install them via:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
Flask>=2.0.0
joblib>=1.0.0
scikit-learn>=1.0.0
nltk>=3.6.0
```

## 📝 NLTK Data

Before running the app, download necessary NLTK corpora:

```bash
python - <<EOF
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
EOF
```

---

## 🚀 Running the App

1. **Clone the repository**:

   ```bash
   git clone https://github.com/PrathamS369/Fake-News-Detection-using-Machine-Learning.git
   cd fake-news-detector
````


````
2. **Set up a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
# or .\venv\Scripts\activate  # Windows
pip install -r requirements.txt
````

3. **Place model artifacts** (`.joblib`) and CSS/templates are already included.

4. **Run the Flask server**:

```
flask run
```

5. Result

---

## 📸 Screenshots

### Home Page

![Home Page](images/screenshot_index.png)

### Prediction Result

![Result Page](images/screenshot_result.png)

---

## 🛠 Troubleshooting

- **Missing NLTK data error**: Ensure you ran the NLTK download commands above.
- **Model load error**: Verify that `tfidf_vectorizer.joblib` and `logreg_fake_news.joblib` are in the root directory.
- **Port in use**: Run `flask run --port=5001` or modify your `launch.json`/environment variable.



## 📄 License

This project is released under the MIT License. See `LICENSE` for details.


