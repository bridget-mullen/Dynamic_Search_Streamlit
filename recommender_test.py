import streamlit as st
import pandas as pd
import faiss
import numpy as np
from joblib import load
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
import lz4.frame
import gdown

# Set page config

st.set_page_config(
    layout="wide",
    page_title="HAM Dynamic Image Search",
    initial_sidebar_state="expanded"
)



# File IDs from Google Drive
GDRIVE_FILES = {
    "csv": ("small_HAM.csv", "1ChFI1o0WqVHZthHp-YwdFutBga_pYAdK"),
    "image_index": ("image_index4.faiss", "1X9rWa84Ve1fZX9AXcIGzt8wjFvYqis7n"),
    "compressed_text_index": ("tfidf_index_6.faiss.lz4", "1g8qCUt4W63PCGU8ShTsJX-4Xgi-hw95-"),
    "joblib": ("tfidf_data_6.joblib", "16x0CpZiFS-JKVWRal6pRTdAJ-HxcWJ9v"),
}

def download_from_gdrive(file_id, dest):
    """Download a file from Google Drive using gdown."""
    try:
        gdown.download(id=file_id, output=dest, quiet=False)
    except Exception as e:
        st.error(f"Download failed for {dest} (ID: {file_id}): {e}")

@st.cache_resource
def download_and_prepare_files():
    for name, (filename, file_id) in GDRIVE_FILES.items():
        if not os.path.exists(filename):
           # st.info(f"Downloading {filename}...")
            try:
                download_from_gdrive(file_id, filename)
              #  st.success(f"Downloaded {filename}")
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
                st.stop()

@st.cache_resource
def decompress_index_if_needed():
    compressed_name = GDRIVE_FILES["compressed_text_index"][0]
    decompressed_name = "tfidf_index_6.faiss"
    
    if not os.path.exists(decompressed_name):
        if not os.path.exists(compressed_name):
            st.error(f"Missing compressed index: {compressed_name}")
            st.stop()
        try:
            with lz4.frame.open(compressed_name, "rb") as f_in:
                with open(decompressed_name, "wb") as f_out:
                    f_out.write(f_in.read())
        except Exception as e:
            st.error(f"Decompression failed: {e}")
            st.stop()
    
    return True

@st.cache_resource
def load_data():
    download_and_prepare_files()
    decompress_index_if_needed()

    df = pd.read_csv(GDRIVE_FILES["csv"][0])
    
    # Clean string columns
    for col in ['title', 'artist_name', 'date']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "" if pd.isna(x) or str(x).lower() == 'nan' else str(x).strip())

    def load_index(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing index file: {path}")
        return faiss.read_index(path, faiss.IO_FLAG_MMAP)

    data = {
        "df": df,
        "image_index": load_index(GDRIVE_FILES["image_index"][0]),
        "text_index": load_index("tfidf_index_6.faiss"),
        "tfidf": load(GDRIVE_FILES["joblib"][0])["vectorizer"]
    }
    
    return data

# Main app class
class HAMRecommendStreamlit:

    def __init__(self):
        self.init_session_state()
        try:
            self.data = load_data()
            self.df = self.data["df"]
            self.display_ui()
        except Exception as e:
            st.error(f"Data load error: {str(e)}")
            st.stop()
        


    def init_session_state(self):
        """Initialize all session state variables"""
        required_states = {
            'current_likes': [],  
            'batches': [],
            'liked_images': {},
            'last_expanded': 0
        }
        
        for key, default in required_states.items():
            if key not in st.session_state:
                st.session_state[key] = default
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def download_image(_self, url):
        """Cache individual image downloads"""
        try:
            response = requests.get(url, timeout=8)
            if response.status_code == 200:
                return BytesIO(response.content).read()
            return None
        except:
            return None

    def download_images_concurrently(self, urls):
        """Download multiple images concurrently using threading"""
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self.download_image, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    results[url] = None
        return results
    

    
    def search_for_text(self, query):
        """Search for and return similar objects based on text query"""
        try:
            # Create query vector
            query_vec = self.data['tfidf'].transform([query])
            #query_vec /= np.linalg.norm(query_vec)
            
            # Get results
            D, I = self.data['text_index'].search(query_vec.toarray().astype("float32"), k=36)  # Get more to account for filtering
            
            new_recommendations = []
            new_scores = []
                
            for idx, score in zip(I[0], D[0]):
                if  score > 0:
                    new_recommendations.append(idx)
                    new_scores.append(score)
                
            self.show_images(new_recommendations[:30], new_scores[:30], 
                               batch_name=f"Search Results for: '{query}'")
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
                
            # Show results
           # self.show_images(I[0], D[0], batch_name=f"Search Results for: '{query}'")
    
    def search_by_likes(self):
        """Get recommendations based ONLY on current likes"""
        if not st.session_state.current_likes:
            st.warning("Please like some images first!")
            return
        
        # Clear previous recommendations
        self.clear_past_recommendations()
        
        try:
            # Use only current likes for recommendations
            current_likes = st.session_state.current_likes
            liked_embeddings = [np.array(eval(self.df.iloc[idx]['image_embedding'])) 
                              for idx in current_likes]
            
            avg_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1).astype('float32')
            
            D, I = self.data['image_index'].search(avg_embedding, k=36)
            
            new_recommendations = []
            new_scores = []
            
            for idx, score in zip(I[0], D[0]):
                if idx not in current_likes and score > 0:
                    new_recommendations.append(idx)
                    new_scores.append(score)
            
            # Filter out already liked images
           # new_recommendations = [idx for idx in I[0] if idx not in current_likes]
            #new_scores = [D[0][i] for i, idx in enumerate(I[0]) if idx not in current_likes]
            
            self.show_images(new_recommendations[:21], new_scores[:21], 
                            batch_name="Recommended Based On Your Current Likes")
            
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            
    def display_liked_items(self):
        """Display liked items in the left sidebar"""
        to_remove = []
        for idx in st.session_state.current_likes:
            if 0 <= idx < len(self.df):
                art = self.df.iloc[idx]
                img_data = st.session_state.liked_images.get(idx)
                
                if img_data:
                    with st.container():
                        st.image(img_data, 
                                caption=f"{art.title[:50]}..." if len(art.title) > 50 else art.title,
                                width=None)
                        if st.button("Remove", key=f"remove_{idx}"):
                            to_remove.append(idx)
        
        # Process removals
        if to_remove:
            for idx in to_remove:
                if idx in st.session_state.current_likes:
                    st.session_state.current_likes.remove(idx)
                if idx in st.session_state.liked_images:
                    del st.session_state.liked_images[idx]
            #st.rerun()
    
    def clear_likes(self):
        """Clear ALL current likes"""
        st.session_state.current_likes = []
        st.session_state.liked_images = {}
        st.success("Cleared all current likes!")
        st.rerun()
        
    def on_like_change(self, idx):
        """Handle like/unlike actions for current session only"""
        if idx in st.session_state.current_likes:
            st.session_state.current_likes.remove(idx)
        else:
            st.session_state.current_likes.append(idx)
            if idx not in st.session_state.liked_images:
                url = self.df.iloc[idx]['primaryimageurl']
                st.session_state.liked_images[idx] = self.download_image(url)
        

        
    def clear_past_recommendations(self):
        """Clear all batches except the original search results"""
        if st.session_state.batches:
            # Keep only the first batch (original search results)
            st.session_state.batches = [st.session_state.batches[0]]
            st.session_state.last_expanded = 0
    
    def show_images(self, indices, scores, batch_name=None):
        """Display images without using historical data"""
        batch_data = {
            "name": batch_name or f"Results {len(st.session_state.batches) + 1}",
            "images": []
        }

        # Download images fresh for this batch
        urls = [self.df.iloc[idx]['primaryimageurl'] for idx in indices]
        downloaded_images = self.download_images_concurrently(urls)
        
        for idx, score, url in zip(indices, scores, urls):
            image_data = downloaded_images[url]
            if image_data:
                art = self.df.iloc[idx]
                raw_info = f"{art.get('title', '')}, {art.get('artist_name', '')}, {art.get('date', '')}"

                # Replace multiple commas or comma+spaces with a single comma
                cleaned_info = re.sub(r'(,\s*)+', ', ', raw_info)
                
                # Remove leading/trailing commas and spaces
                cleaned_info = cleaned_info.strip(', ').strip()
                
             
              
                batch_data["images"].append({
                    "idx": idx,
                    "score": score,
                    "image_data": image_data,
                    "info": cleaned_info,
                    "liked": idx in st.session_state.current_likes
                })
        
        if batch_data["images"]:
            st.session_state.batches.append(batch_data)
            st.session_state.last_expanded = len(st.session_state.batches) - 1
    
    def display_ui(self):
        """Display the UI with left sidebar"""
        # Use Streamlit's native sidebar for the likes panel
        with st.sidebar:
            st.header("Your Likes ❤️")
            if st.session_state.current_likes:
                st.write(f"{len(st.session_state.current_likes)} liked artworks")
                self.display_liked_items()
            else:
                st.write("No liked artworks yet")

        # Main content area
        self.display_main_content()
            
            
    def display_main_content(self):
        """Display main content with buttons"""
        st.title("HAM Dynamic Image Search")
        
        # Custom CSS for button styling
        st.markdown("""
        <style>
            .stButton>button {
                border: 2px solid #4CAF50;
                color: white;
                background-color: #4CAF50;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #45a049;
                border-color: #45a049;
            }
            .recommend-button {
                background-color: #2196F3 !important;
                border-color: #2196F3 !important;
            }
            .recommend-button:hover {
                background-color: #0b7dda !important;
                border-color: #0b7dda !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
        # Search row with inline button
        search_col, button_col = st.columns([4, 1])
        with search_col:
            query = st.text_input("Search for art:", label_visibility="collapsed", placeholder="Enter search terms...")
        with button_col:
            if st.button("🔍 Find Images", key="find_images"):
                if query:
                    st.session_state.last_expanded = len(st.session_state.batches)
                    self.search_for_text(query)
                else:
                    st.warning("Please enter search terms")
    
        # Recommendation button with custom styling
        if st.button("✨ Find Images Similar to Likes", 
                    disabled=not st.session_state.current_likes,
                    key="get_recommendations"):
            with st.spinner("Finding images..."):
                st.session_state.last_expanded = len(st.session_state.batches)
                self.search_by_likes()
        
        # Clear button
        if st.button("Clear All Likes", type="secondary"):
            self.clear_likes()
    
        # Display batches
        for batch_idx, batch in enumerate(st.session_state.batches):
            # Create container for the header and expander
            batch_container = st.container()
            
            with batch_container:
                # header above the expander
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin: 10px 0;">
                        {batch["name"]}
                        <span style="margin-left: 8px; font-size: 0.8em; color: #666;">
                            {'▼' if batch_idx == st.session_state.last_expanded else '▶'}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Expander with content
                with st.expander("", expanded=batch_idx == st.session_state.last_expanded):
                    cols = st.columns(3)
                    for i, img in enumerate(batch["images"]):
                        with cols[i % 3]:
                            # Custom card container with subtle border and padding
                            st.markdown(
                                """
                                <style>
                                    .art-card {
                                        border: 1px solid #e0e0e0;
                                        border-radius: 8px;
                                        padding: 12px;
                                        margin-bottom: 16px;
                                        background-color: #f9f9f9;
                                        transition: all 0.2s ease;
                                    }
                                    .art-card:hover {
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                        background-color: #f5f5f5;
                                    }
                                    .like-container {
                                        display: flex;
                                        justify-content: center;
                                        margin-top: 8px;
                                        padding-top: 8px;
                                        border-top: 1px solid #eee;
                                     }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Card container
                            with st.container():
                                st.markdown('<div class="art-card">', unsafe_allow_html=True)
                                
                                # Image
                                st.image(
                                    img["image_data"], 
                                    width=None,
                                    output_format="auto"
                                )
                                
                                # Artwork info
                                st.write(img["info"])
                                
                                # Like button container
                                st.markdown('<div class="like-container">', unsafe_allow_html=True)
                                st.checkbox(
                                    "❤️ Like this artwork",  # More descriptive label
                                    value=img["idx"] in st.session_state.current_likes,
                                    #key=f"like_{batch_idx}_{img['idx']}",
                                    key=f"like_{batch['name']}_{img['idx']}_{batch_idx}_{i}",  # Unique key
                                    on_change=self.on_like_change,
                                    args=(img["idx"],),
                                    label_visibility="visible"
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                

    def display_sidebar(self):
        """Display the sidebar content"""
        st.header("Your Likes")
        st.write(f"Liked artworks: {len(st.session_state.current_likes)}")  # Updated reference
        
        to_remove = []
        for idx in st.session_state.current_likes:  # Updated reference
            if 0 <= idx < len(self.df):
                art = self.df.iloc[idx]
                img_data = st.session_state.liked_images.get(idx)
                
                if img_data:
                    st.image(img_data, caption=art['title'], width=None)
                    if st.button("Remove", key=f"remove_{idx}"):
                        to_remove.append(idx)
                else:
                    st.error(f"Couldn't load image for {art.title}")
            else:
                to_remove.append(idx)
        
        if to_remove:
            for idx in to_remove:
                if idx in st.session_state.current_likes:  # Updated reference
                    st.session_state.current_likes.remove(idx)
                if idx in st.session_state.liked_images:
                    del st.session_state.liked_images[idx]
            st.rerun()

# Run the app
if __name__ == "__main__":
    app = HAMRecommendStreamlit()