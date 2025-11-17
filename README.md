🎬 Movie Recommendation System

A simple and clean machine-learning project that recommends movies based on similarity using NLP.

🛠 Tech Stack

Python 3.x

scikit-learn

pandas, numpy

Streamlit (optional)

📂 Project Structure
Movie-Recommendation-System/
│── data/
│── src/
│── app.py
│── requirements.txt
│── README.md

🧠 How It Works (Short)
1️⃣ Prepare Dataset

Merge movie & credits data and keep the useful columns.

2️⃣ Create a “tags” column
df["tags"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

3️⃣ Convert Text → Numbers
from sklearn.feature_extraction.text import CountVectorizer
vectors = CountVectorizer(stop_words="english").fit_transform(df["tags"]).toarray()

4️⃣ Compute Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

▶️ How to Run

Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


(Optional)

streamlit run app.py

⭐ Future Updates

Add posters

Improve UI

Add more recommendation logic
