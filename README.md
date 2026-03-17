# CineNext-Recommender 🎬

Moving beyond generative AI to explore deterministic recommendation engines using Cosine Similarity and Vectorization.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cinenext-recommender.streamlit.app/)

## 🚀 Live Demo

**Try the Recommender App here:** [CineNext on Streamlit](https://cinenext-recommender.streamlit.app/)

Search for specific genres, vibes, or movie titles. The engine uses vectorization via ChromaDB to find the closest semantic matches to your query.

### Examples in Action

**1. Semantic Search:** Search for "Korean" movies, and the AI searches for the most similar meaning to the word 'Korean' using vectorization.
![Korean Search Example 1](https://github.com/user-attachments/assets/0e2a0241-00a7-41e6-a613-273f03f7f22f)

<br>

![Korean Search Example 2](https://github.com/user-attachments/assets/0adffc9c-44cc-45f6-b8aa-d28023e08a39)

<br>

*(Note: No actual movies will be played or streamed from this app.)*
![Player Example](https://github.com/user-attachments/assets/30a687e9-f50f-4e73-a0ef-7fd2cb11e06f)

<br>

**2. Vibe-Based Recommendations:** Type a specific movie like *Inception*, or simply type "funny", and it will return recommendations based on that exact vibe. 
![Inception Example](https://github.com/user-attachments/assets/3455d013-544b-439a-b260-47fd7ed94429)

<br>

**3. Deep Dives:** Use the "Show more results" button to display up to 50 similar movies.
![Show More Results](https://github.com/user-attachments/assets/71921b15-e12f-435e-a4e3-0714437f61a2)

---

## 🛠️ Built With

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** Sentence-Transformers
* **Language:** Python 

---

## 📂 Project Structure

* `app.py`: The main Streamlit application script handling the user interface and search logic.
* `movie_list.pkl`: Pre-processed movie dataset.
* `Colab Files/`: Jupyter Notebooks containing the data cleaning, processing, and vector embedding generation.
* `requirements.txt`: Python dependencies required to run the application.

---

## 💻 How to Run Locally

If you want to run this project on your own machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/popolome/CineNext-Recommender.git](https://github.com/popolome/CineNext-Recommender.git)
   cd CineNext-Recommender
2. **Install the dependencies:**
   <br>
    It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

---

## 📄 License
This project is licensed under the [MIT License](https://www.google.com/search?q=MIT-LICENSE).
