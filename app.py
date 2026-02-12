# Imports
import streamlit as st
import chromadb
import pandas as pd
import re
import pickle

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
        metadatas=[{'title': t} for t in movies['title'].tolist()],
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

# This is the UI (User Interface) design
st.title("CineNext: AI Movie Recommender 🍿")
st.markdown("Discover movies using titles or just describe what you're looking for.")

user_input = st.text_input("Search movie title or describe a vibe...", placeholder="e.g. Inception or 'A sad movie about robots")

if st.button('Get Recommendations'):
  if user_input:
    with st.spinner('Thinking...'):
      # This will check if the input matches a title
      normalized_input = normalize(user_input)
      match = movies[movies['title'].apply(normalize) == normalized_input]

      if not match.empty:
        # This will search using its tags to if title is found
        query_text = match['tags'].values[0]
        st.write(f"### Recommendations for you, starting with **{match['title'].values[0]}**:")
      else:
        # This will search using the raw description if title is not found
        query_text = user_input
        st.write(f"### We couldn't find that movie, but you might enjoy these similar titles: '{user_input}'")

      # This will query the ChromaDB
      results = collection.query(
        query_texts=[query_text],
        n_results=5
      )

      # This will display the results
      st.divider()
      for res in results['metadatas'][0]:
        st.subheader(f"🎬 {res['title']}")
  else:
    st.warning(f"Please enter something first!")

# This is the footer
st.markdown("---")
st.caption("Powered by ChromaDB & Sentence-Transformers")
