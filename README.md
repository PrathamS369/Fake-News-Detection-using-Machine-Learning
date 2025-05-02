# Fake News Detection Flask App

A simple web application that classifies news articles as **Real** or **Fake** using a Machine Learning pipeline (TF‑IDF + Logistic Regression) and serves predictions via a Flask interface.

---

## 🧰 Project Structure

```
fake_news_app/
├── app.py                     # Main Flask application
├── logreg_fake_news.joblib    # Trained Logistic Regression model
├── tfidf_vectorizer.joblib    # Fitted TF-IDF vectorizer
├── static/
│   └── style.css              # CSS for the frontend
├── templates/
│   ├── index.html             # Home page
│   └── result.html            # Result display page
├── Screenshots/               # App UI screenshots
│   ├── input_1.png
│   ├── input_2.png
│   ├── result_1.png
│   └── result_2.png
├── requirements.txt           # List of Python dependencies
├── launch.json                # Optional VS Code configuration
├── .gitignore                 # To ignore unnecessary files in version control
```

* **app.py**: Main Flask application.
* **.joblib** files: Serialised TF‑IDF vectorizer and trained Logistic Regression model.
* **static/style.css**: Custom CSS for styling.
* **templates/**: HTML templates for input form and result page.
* **screenshots/**: Screenshots demonstrating the UI.
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

Before running the app, download the necessary NLTK corpora:

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
   ```


2. **Set up a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
.\venv\Scripts\activate  # Windows
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

![Home Page](screenshots/input_1)
![Home Page](screenshots/input_2)

### Prediction Result

![Result Page](screenshots/result_1)
![Result Page](screenshots/result_2)

---

## 🛠 Troubleshooting

- **Missing NLTK data error**: Ensure you ran the NLTK download commands above.
- **Model load error**: Verify that `tfidf_vectorizer.joblib` and `logreg_fake_news.joblib` are in the root directory.
- **Port in use**: Run `flask run --port=5001` or modify your `launch.json`/environment variable.



## 📄 License

This project is released under the MIT License. See `LICENSE` for details.


