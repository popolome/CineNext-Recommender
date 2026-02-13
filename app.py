# Imports
import streamlit as st
import chromadb
import pandas as pd
import re
import pickle
import requests

# This is the page config
st.set_page_config(page_title="CineNext AI", page_icon='🍿', layout='wide')

# This is the CSS
st.markdown("""
  <style>
  /* This will ensure posters have rounded corners and a shadow */
  .stImage img {
    border-radius: 10px;
    transition: transform .2s; /* Animation! */
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.5);
  }
  /* This will zoom in slightly when hovering */
  .stImage img:hover {
    transform: scale(1.05);
    cursor: pointer;
  }
  /* This will darken the background for cinematic feel */
  .stApp {
    background-color: #0e1117;
  }
  </style>
""", unsafe_allow_html=True)

# This will load the data fast by using cache
@st.cache_resource
def init_db():
  # This will connect to the chroma folder
  # This will try to get the collection and check if its empty, rebuilts it if empty
  client = chromadb.PersistentClient(path="./movie_db")
  collection = client.get_or_create_collection(name="movies_v2")
  movies = pickle.load(open("movie_list.pkl", 'rb'))

  if collection.count() == 0:
    with st.spinner("Building AI database..."):
      collection.add(
        documents=movies['tags'].tolist(),
        metadatas=[{'title': t, 'id': i} for t, i in zip(movies['title'], movies['id'])],
        ids=[str(i) for i in movies['id'].tolist()]
      )
  return collection, movies

collection, movies = init_db()
# This will run st.toast once
if 'toast_shown' not in st.session_state:
  st.toast("CineNext Engine is Ready!", icon="🍿")
  st.session_state['toast_shown'] = True

# This will initialize the state
if 'display_limit' not in st.session_state:
  st.session_state.display_limit = 10

# This will lock the active search results
if 'search_active' not in st.session_state:
  st.session_state.search_active = False

# This will normalize the logic with re
def normalize(text):
  return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def fetch_details(movie_id):
  # This keeps the API key hidden
  api_key = st.secrets["TMDB_API_KEY"]
  url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-us"
  try:
    data = requests.get(url, timeout=5).json()
    return {
      "poster": "https://image.tmdb.org/t/p/w500" + data.get('poster_path', ''),
      "overview": data.get('overview', 'No description available.'),
      "rating": round(data.get('vote_average', 0), 1)
    }
  except:
    return {
      "poster": "https://via.placeholder.com/500x750?text=No+Poster",
      "overview": "Information unavailable.",
      "rating": "N/A"
    }

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
        n_results=st.session_state.display_limit
      )
      
      st.divider()

      movies_found = results['metadatas'][0]

      # This will loop thru the movies in chunks of 5
      for i in range(0, len(movies_found), 5):
        cols = st.columns(5)
        batch = movies_found[i : i+5]

      # This will use the metadata id to get poster
        for idx, res in enumerate(batch):
          with cols[idx]:
            # This will fetch all details at once, show the poster, and add the interactive popover
            details = fetch_details(res['id'])
            st.image(details['poster'], use_container_width=True)
            # This displays the title in a nice clean font
            with st.popover(f"📖 Details"):
              st.write(f"### {res['title']}")
              st.write(f"⭐ **Rating:** {details['rating']}/10")
              st.write(details['overview'])

      if len(movies_found) >= st.session_state.display_limit and st.session_state.display_limit < 50:
        if st.button("Show More Results ⬇️", key="show_more_btn"):
          st.session_state.display_limit += 10
          st.rerun()
      
  else:
    st.warning(f"Please enter something first!")

# This is the UI (User Interface) design
st.title("CineNext: AI Movie Recommender 🍿")
st.markdown("Discover movies using titles or just describe what you're looking for.")

# This is the search logic for the enter button
user_input = st.text_input("Search movie title or describe a vibe...", placeholder="e.g. Inception or 'A sad movie about robots")

if 'last_query' not in st.session_state:
  st.session_state.last_query = ""

# This only runs if button or enter is pressed, and resets limit for every new search
if st.button('Get Recommendations') or (user_input != st.session_state.last_query and user_input != ""):
  if user_input.strip() != "":
    st.session_state.search_active = True
    st.session_state.display_limit = 10
    st.session_state.last_query = user_input
  else:
    st.warning("Please enter something first!")

# This is the persistent view
if st.session_state.search_active:
  run_recommendation()

# This is the footer
st.markdown("---")
st.caption("Powered by ChromaDB & Sentence-Transformers. Data provided by TMDB.")
st.caption("CineNext Recommender v1.0")
