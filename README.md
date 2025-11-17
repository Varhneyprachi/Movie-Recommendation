🎬 Movie Recommendation System

A machine learning–powered movie recommender built using Python, Pandas, NumPy, Scikit-Learn, and NLTK.
This project suggests movies based on content similarity, helping users discover films similar to their favorites.

🚀 Features

🔍 Search any movie and instantly get similar recommendations

🤖 Content-based filtering using NLP & cosine similarity

🧹 Data preprocessing, text cleaning, and feature engineering

📊 Uses TMDB dataset with genres, keywords, cast, and crew

⚡ Fast and highly scalable similarity computation

📝 Clean, readable Python code

🧠 Tech Stack
Category	Technology
Language	Python
ML / NLP	Scikit-learn, NLTK
Data	Pandas, NumPy
Visualization	Matplotlib, Seaborn (optional)
📁 Project Structure
Movie-Recommendation/
│── data/
│   ├── movies.csv
│   ├── credits.csv
│
│── notebooks/
│   ├── EDA.ipynb
│   ├── model_building.ipynb
│
│── src/
│   ├── recommend.py
│   ├── preprocessing.py
│   ├── utils.py
│
│── app.py
│── requirements.txt
│── README.md


(Folders may vary depending on your exact project)

⚙️ How It Works
1️⃣ Load & Merge Datasets
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

df = movies.merge(credits, on="id")

2️⃣ Clean & Prepare Text Data
df['tags'] = df['overview'] + " " + df['genres'] + " " + df['keywords']
df['tags'] = df['tags'].apply(lambda x: x.lower())

3️⃣ Convert Tags → Vectors
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

4️⃣ Compute Similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

5️⃣ Recommend Function
def recommend(movie):
    index = df[df['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), 
                        reverse=True, 
                        key=lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(df.iloc[i[0]].title)

▶️ Running the Project
Install dependencies
pip install -r requirements.txt

Run the recommender
python app.py

📸 Sample Output
Enter movie: Avatar

Top Recommendations:
1. Guardians of the Galaxy
2. Star Trek
3. John Carter
4. The Avengers
5. Star Wars

🔮 Future Enhancements

✔ Add user-based collaborative filtering

✔ Deploy as a web app (Flask/Streamlit)

✔ Add posters & movie metadata via TMDB API

✔ Build a hybrid recommender

🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to improve.

⭐ Show Your Support

If you find this helpful, don’t forget to ⭐ star the repository!
