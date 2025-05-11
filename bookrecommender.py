import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

numpy._import_array()

# --- Load Data ---
@st.cache_data
def load_data():
    df_books = pd.read_csv("cleaned_books.csv")
    df_ratings = pd.read_csv("cleaned_ratings.csv")
    df_tags = pd.read_csv("cleaned_book_tags.csv")
    return df_books, df_ratings, df_tags

df_books, df_ratings, df_tags = load_data()

# --- Helper for displaying images ---
def render_table_with_images(df, image_column, columns_to_show):
    df = df.copy()
    df[image_column] = df[image_column].apply(lambda url: f'<img src="{url}" width="50">')
    st.write(
        df[columns_to_show].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    
# --- Content-Based Functions ---
@st.cache_data
def create_tfidf_matrix():
    authors_title_df = df_books[['book_id', 'original_title', 'title', 'authors', 'average_rating', 'image_url']].copy()
    authors_title_df['content'] = authors_title_df['original_title'].fillna('') + ' ' + authors_title_df['authors'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(authors_title_df['content'])
    return tfidf_matrix, authors_title_df

def rcmd_content_based(ori_book_id, k=5):
    tfidf_matrix, authors_title_df = create_tfidf_matrix()
    idx_series = df_books.index[df_books['book_id'] == ori_book_id]
    if len(idx_series) == 0:
        return pd.DataFrame()
    idx = idx_series[0]
    v = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(v[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    book_indices = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]
    rcmd = authors_title_df.iloc[book_indices][["book_id", "title", "authors", "average_rating", "image_url"]].copy()
    rcmd['similarity_score'] = similarity_values
    rcmd = rcmd.reset_index(drop=True)
    return rcmd

# --- Collaborative Filtering Functions ---
@st.cache_resource
def train_svd_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[['user_id', 'book_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=20)
    model.fit(trainset)
    return model

def get_top_n_cf(model, user_id, n=5):
    user_books = set(df_ratings[df_ratings['user_id'] == user_id]['book_id'])
    all_books = set(df_books['book_id'])
    books_to_predict = list(all_books - user_books)
    predictions = []
    for book_id in books_to_predict:
        pred = model.predict(user_id, book_id)
        predictions.append((book_id, pred.est, pred.details.get('was_impossible', False)))
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    df_top_n = pd.DataFrame(top_n, columns=['book_id', 'predicted_rating', 'was_impossible'])
    # Merge with book info
    df_top_n = df_top_n.merge(df_books[['book_id', 'title', 'authors', 'image_url']], on='book_id', how='left')
    # Merge with actual rating if it exists (should be NaN for unseen books)
    df_top_n = df_top_n.merge(
        df_ratings[df_ratings['user_id'] == user_id][['book_id', 'rating']],
        on='book_id', how='left'
    )
    df_top_n.rename(columns={'rating': 'actual_rating'}, inplace=True)
    return df_top_n

# --- Streamlit UI ---
st.title("Book Recommender System")

tab1, tab2 = st.tabs(["Collaborative Filtering", "Content-Based"])

with tab1:
    st.header("Collaborative Filtering (SVD)")
    user_ids = df_ratings['user_id'].unique()
    user_id = st.selectbox("Select User ID", sorted(user_ids))
    n_cf = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Get Recommendations", key="cf"):
        model = train_svd_model()
        df_top_n = get_top_n_cf(model, user_id, n_cf)
        st.write(f"Top {n_cf} recommendations for User {user_id}:")
        render_table_with_images(
            df_top_n,
            'image_url',
            ['title', 'authors', 'predicted_rating', 'image_url']
        )

with tab2:
    st.header("Content-Based Filtering")
    book_titles = df_books[['book_id', 'title']].drop_duplicates().sort_values('title')
    book_title = st.selectbox("Select a Book", book_titles['title'])
    book_id = book_titles[book_titles['title'] == book_title]['book_id'].iloc[0]
    n_cb = st.slider("Number of Recommendations", 1, 10, 5, key="cb_slider")
    if st.button("Get Recommendations", key="cb"):
        rcmd = rcmd_content_based(book_id, n_cb)
        st.write(f"Top {n_cb} similar books to '{book_title}':")
        render_table_with_images(
            rcmd,
            'image_url',
            ['title', 'authors', 'average_rating', 'similarity_score', 'image_url']
        )
