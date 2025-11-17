🎬 Movie Recommendation System – Machine Learning Project

A content-based movie recommender built using Python, NLP, and Machine Learning.
The system recommends movies similar to the one you select — using text vectorization + cosine similarity.

🌟 Highlights

🎯 Accurate content-based recommendations

🤖 Uses NLP on genres, keywords, overview, cast & crew

⚡ Fast cosine similarity search on 5000+ feature vectors

🧹 Complete preprocessing pipeline

🖥 Can be deployed via Streamlit

🗂 Clean modular Python code

🛠 Tech Used
Component	Technology
Language	Python 3.x
ML / NLP	scikit-learn, NLTK
Data	pandas, numpy
Deployment	Streamlit (optional)
📂 Project Structure
Movie-Recommendation-System/
│── data/
│   ├── movies.csv
│   ├── credits.csv
│
│── src/
│   ├── preprocess.py
│   ├── recommend.py
│   ├── utils.py
│
│── app.py
│── requirements.txt
│── README.md

🧠 How the Recommender Works

A simple overview of the exact ML logic used.

1️⃣ Load & Merge Data
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

df = movies.merge(credits, on="id")

2️⃣ Clean + Combine Features Into a Single “Tags” Column
def clean_text(x):
    return x.lower().replace(" ", "")

df["tags"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]
df["tags"] = df["tags"].apply(clean_text)

3️⃣ Vectorization Using CountVectorizer (Top 5000 Words)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

4️⃣ Similarity Matrix (Cosine Similarity)
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

5️⃣ Recommendation Function
def recommend(movie_name):
    index = df[df["title"] == movie_name].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    for i in movies_list:
        print(df.iloc[i[0]].title)

▶️ Run Locally
Install dependencies
pip install -r requirements.txt

Start the project
python app.py

(Optional) Run the Streamlit App
streamlit run app.py

📸 Example Output
Enter movie: Inception

Recommended Movies:
1. Interstellar
2. Shutter Island
3. The Prestige
4. The Matrix
5. Tenet

🚀 Future Improvements

Add TMDB API for posters & movie details

Add collaborative filtering (user-based)

Add hybrid recommendation engine

Add rating-based ranking

Deploy on Render/Netlify/Streamlit Cloud

🤝 Contributing

Feel free to open PRs or issues. Contributions are always welcome!

⭐ Show Support

If this project helped you, please ⭐ the repo!

