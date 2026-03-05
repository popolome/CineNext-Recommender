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
 /* This is the Cinematic Background */
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

@st.cache_data(ttl=86400)
def fetch_details(movie_id):
  # This keeps the API key hidden
  api_key = st.secrets["TMDB_API_KEY"]
  url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-us&append_to_response=videos"
  try:
    data = requests.get(url, timeout=5).json()

    # This will format the year
    release_date = data.get('release_date', '')
    year = release_date.split('-')[0] if release_date else 'N/A'

    # This will format the runtime (e.g., 125 -> 2h 5m)
    runtime = data.get('runtime', 0)
    runtime_str = f"{runtime // 60}hr {runtime % 60}mins" if runtime else 'N/A'

    # This will extract up to 3 genres
    genres = [g['name'] for g in data.get('genres', [])][:3]

    # This will get the wide back-drop image, else use a poster
    backdrop_path = data.get('backdrop_path')
    backdrop = f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else "https://image.tmdb.org/t/p/w500" + data.get('poster_path', '')

    # This will loop and find trailers from Youtube and stop once it found it
    trailer_key = None
    videos = data.get('videos', {}).get('results', [])
    for vid in videos:
      if vid.get('site') == 'YouTube' and vid.get('type') == 'Trailer':
        trailer_key = vid.get('key')
        break
    
    return {
      "poster": "https://image.tmdb.org/t/p/w500" + data.get('poster_path', ''),
      "backdrop": backdrop,
      "overview": data.get('overview', 'No description available.'),
      "rating": round(data.get('vote_average', 0), 1),
      "year": year,
      "runtime": runtime_str,
      "genres": genres
    }
  except:
    return {
      "poster": "https://via.placeholder.com/500x750?text=No+Poster",
      "backdrop": "https://via.placeholder.com/1280x720?text=No+Image",
      "overview": "Information unavailable.",
      "rating": "N/A",
      "year": "N/A",
      "runtime": "N/A",
      "genres": []
    }

@st.dialog(" ")
def show_details(movie_id, title):
  details = fetch_details(movie_id)

  # This will loop through the genres to build the grey badges dynamically
  genres_html = "".join([f"<span class='badge'>{g}</span>" for g in details['genres']])

  st.markdown(f"""
    <style>
    /* This will pull the banner image up closer to the close button */
    div[data-testid="stDialog"] div[data-testid="stVerticalBlock"] > div:first-child {{
      margin-top: -1.5rem;
    }}

    .hero-banner {{
      position: relative;
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 20px;
    }}

    .hero-image {{
      height: 350px;
      background-image: url('{details['backdrop']}');
      background-size: cover;
      background-position: center top;
    }}

    .hero-gradient {{
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 80%;
      background: linear-gradient(to top, #262730 5%, transparent);
    }}

    .hero-title {{
      position: absolute;
      bottom: 15px;
      left: 20px;
      font-size: 3rem;
      font-weight: 800;
      text-transform: uppercase;
      margin: 0;
      color: white;
      line-height: 1.1;
      text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    }}

    .badge-container {{
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}

    .badge {{
      background-color: #444;
      padding: 5px 12px;
      border-radius: 4px;
      font-size: 0.85rem;
      font-weight: 600;
      color: white;
    }}

    .overview-text {{
      font-size: 1.15rem;
      line-height: 1.6;
      color: #e5e5e5;
      margin-bottom: 30px;
    }}

    div[data-testid="stDialog"] div[data-testid="stButton"] > button {{
      background-color: #e50914 !important;
      color: white !important;
      border-radius: 5px !important;
      width: 160px !important;
      height: 45px !important;
      border: none !important;
      box-shadow: 0 4px 10px rgba(0,0,0,0.4) !important;
    }}

    div[data-testid="stDialog"] div[data-testid="stButton"] > button p {{
      display: block !important;
      font-size: 1.1rem !important;
      font-weight: bold !important;
      margin: 0 !important;
    }}

    div[data-testid="stDialog"] div[data-testid="stButton"] > button:hover {{
      background-color: #f40612 !important;
      transform: scale(1.05) !important;
    }}
    </style>

    <div class="hero-banner">
      <div class="hero-image"></div>
      <div class="hero-gradient"></div>
      <div class="hero-title">{title}</div>
    </div>

    <div class="badge-container">
      <span class="badge">{details['year']}</span>
      <span class="badge">⭐ {details['rating']}</span>
      <span class="badge">{details['runtime']}</span>
      {genres_html}
    </div>

    <div class="overview-text">
      {details['overview']}
    </div>
  """, unsafe_allow_html=True)

  trailer = details.get('trailer_key')
  if trailer:
    st.video(f"https://www.youtube.com/watch?v={trailer}")

  # This will add the Netflix-liked button
  if st.button("Play Movie ▶"):
    st.toast(f"This is a test, no video will appear.")
    st.toast(f"Starting {title}...", icon="🎬")

def run_recommendation():
  if user_input:
    # This will do the heavy AI search only if we typed something new, or clicked "Show More"
    is_new_search = st.session_state.get('loaded_query') != user_input
    is_new_limit = st.session_state.get('loaded_limit') != st.session_state.display_limit

    if is_new_search or is_new_limit or "movies_found" not in st.session_state:
      
      with st.spinner('Thinking...'):
        # This will check if the input matches a title
        normalized_input = normalize(user_input)
        match = movies[movies['title'].apply(normalize) == normalized_input]
  
        if not match.empty:
          # This will search using its tags to if title is found
          query_text = match['tags'].values[0]
          st.session_state['header_text'] = f"### Keeping the **{match['title'].values[0]}** vibe going with these picks:"
        else:
          # This will search using the raw description if title is not found
          query_text = user_input
          st.session_state['header_text'] = f"### We couldn't find that movie, but you might enjoy these similar titles:"
  
        # This will query the ChromaDB
        results = collection.query(
          query_texts=[query_text],
          n_results=st.session_state.display_limit
        )

        # This will save the results directly into Streamlit's memory so it will load fast
        st.session_state['movies_found'] = results['metadatas'][0]
        st.session_state['loaded_query'] = user_input
        st.session_state['loaded_limit'] = st.session_state.display_limit

    st.write(st.session_state['header_text'])
    st.divider()

    movies_found = st.session_state['movies_found']

      # This will loop thru the movies in chunks of 5
    for i in range(0, len(movies_found), 5):
      cols = st.columns(5)
      batch = movies_found[i : i+5]

      # This will use the metadata id to get poster
      for idx, res in enumerate(batch):
        with cols[idx]:
          # This will fetch all details at once, show the poster, and add the interactive popover
          details = fetch_details(res['id'])

          st.markdown(f"""
            <div id="movie_{res['id']}"></div>
            <style>
            /* This finds the wrapper containing the anchor, then targets the next button in the next wrapper */
            div.element-container:has(#movie_{res['id']}) + div.element-container button {{
              background-image: url('{details['poster']}');
              background-size: cover;
              background-position: center;
              height: 350px;
              width: 100% !important;
              border-radius: 10px;
              border: 2px solid transparent;
              box-shadow: 0 4px 10px rgba(0,0,0,0.4);
              transition: transform 0.3s ease-in-out, border-color 0.3s ease-in-out;
            }}

            /* This is the hover state */
            div.element-container:has(#movie_{res['id']}) + div.element-container button:hover {{
              transform: scale(1.05);
              border-color: #e50914;
              z-index: 10;
            }}
            </style>
          """, unsafe_allow_html=True)

          # This button will mimic clicking the poster
          if st.button(" ", key=f"btn_{res['id']}", use_container_width=True):
            show_details(res['id'], res['title'])

          # This will show the title below the poster
          st.caption(f"**{res['title']}**")

    if len(movies_found) >= st.session_state.display_limit and st.session_state.display_limit < 50:
      st.divider()
      if st.button("Show More Results ⬇️", key="show_more_btn", use_container_width=True):
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
