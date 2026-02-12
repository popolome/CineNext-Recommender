# Imports
import streamlit as st
import chromadb
import pandas as pd
import re
import pickle
import requests

# This is the page config
st.set_page_config(page_title="CineNext AI", page_icon='🍿', layout='centered')

# This will load the data fast by using cache
@st.cache_resource
def init_db():
  # This will connect to the chroma folder
  # This will try to get the collection and check if its empty, rebuilts it if empty
  client = chromadb.PersistentClient(path="./movie_db")
  collection = client.get_or_create_collection(name="movies")
  movies = pickle.load(open("movie_list.pkl", 'rb'))

  if collection.count() == 0:
    with st.spinner("Building AI database..."):
      collection.add(
        documents=movies['tags'].tolist(),
        metadatas=[{'title': t, 'id': i} for t, i in zip(movies['titile'], movies['id']),
        ids=[str(i) for i in movies['id'].tolist()]
      )
  return collection, movies

collection, movies = init_db()
# This will run st.toast once
if 'toast_shown' not in st.session_state:
  st.toast("CineNext Engine is Ready!", icon="🍿")
  st.session_state['toast_shown'] = True

# This will normalize the logic with re
def normalize(text):
  return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def fetch_poster(movie_id):
  # This keeps the API key hidden
  api_key = st.secrets["TMDB_API_KEY"]
  url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-us"
  try:
    data = requests.get(url, timeout=5).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path
  except:
    return "https://via.placeholder.com/500x750?text=No+Poster+Found"

def run_recommendation():
  if user_input:
    with st.spinner('Thinking...'):
      # This will check if the input matches a title
      normalized_input = normalize(user_input)
      match = movies[movies['title'].apply(normalize) == normalized_input]

      if not match.empty:
        # This will search using its tags to if title is found
        query_text = match['tags'].values[0]
        st.write(f"### Keeping the **{match['title'].values[0]}** vibe going with these picks:")
      else:
        # This will search using the raw description if title is not found
        query_text = user_input
        st.write(f"### We couldn't find that movie, but you might enjoy these similar titles:")

      # This will query the ChromaDB
      results = collection.query(
        query_texts=[query_text],
        n_results=5
      )
      
      st.divider()

      # This will display the results in 5 columns
      cols = st.columns(5)

      # This will use the metadata id to get poster
      for idx, res in enumerate(results['metadatas'][0]):
        with cols[idx]:
          poster_url = fetch_poster(res['id'])
          st.image(poster_url, use_container_width=True)
          # This displays the title in a nice clean font
          st.markdown(f"**{res['title']}**")
      
  else:
    st.warning(f"Please enter something first!")

# This is the UI (User Interface) design
st.title("CineNext: AI Movie Recommender 🍿")
st.markdown("Discover movies using titles or just describe what you're looking for.")

# This is the search logic for the enter button
user_input = st.text_input("Search movie title or describe a vibe...", placeholder="e.g. Inception or 'A sad movie about robots")

# This only runs if button or enter is pressed
if st.button('Get Recommendations') or user_input:
  if user_input.strip() != "":
    run_recommendation()

# This is the footer
st.markdown("---")
st.caption("Powered by ChromaDB & Sentence-Transformers. Data provided by TMDB.")
