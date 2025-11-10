from flask import Flask, render_template, request
import joblib, os, re, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    tokens = text.split()

    # Keep important emotion words
    important_words = {"hate", "love", "angry", "kill", "stupid", "idiot"}
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens 
                       if (word not in stop_words) or (word in important_words)]
    return " ".join(filtered_tokens)

# Auto-train if model not found
if not (os.path.exists("toxic_model.pkl") and os.path.exists("vectorizer.pkl")):
    print("⚙️ Training new model...")
    data = pd.read_csv("youtoxic_english_1000.csv")

    # Combine all toxicity-related flags into one target
    data["target"] = data[["IsToxic","IsAbusive","IsThreat","IsProvocative",
                           "IsObscene","IsHatespeech","IsRacist","IsNationalist",
                           "IsSexist","IsHomophobic","IsReligiousHate",
                           "IsRadicalism"]].any(axis=1).astype(int)

    X = data["Text"]
    y = data["target"]

    vectorizer = TfidfVectorizer(max_features=500)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    joblib.dump(model, "toxic_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("✅ Model trained successfully!")

# Load trained model
model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    comment = request.form["comment"]
    cleaned = clean_text(comment)
    X = vectorizer.transform([cleaned])
    prob = model.predict_proba(X)[0][1]
    threshold = 0.5
    label = "Toxic 😡" if prob >= threshold else "Non-Toxic 😊"

    return render_template(
        "index.html",
        comment=comment,
        prediction=label,
        probability=f"{prob*100:.2f}%",
        cleaned=cleaned
    )

if __name__ == "__main__":
    app.run(debug=True)
