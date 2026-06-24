"""
Anime Recommendation System — Streamlit App
Applied Data Science, ML & AI | E&ICT Academy, IIT Guwahati
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from collections import Counter

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS + Naruto Background ────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(rgba(5,13,26,0.87), rgba(5,13,26,0.87)),
            url("https://images4.alphacoders.com/134/thumb-1920-1340305.jpg");
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-image: radial-gradient(circle,
            rgba(100,160,255,0.05) 1px, transparent 1px);
        background-size: 30px 30px;
        pointer-events: none; z-index: 0;
    }
    [data-testid="stHeader"]         { background: transparent !important; }
    [data-testid="collapsedControl"] { display: none; }
    .block-container {
        padding: 0.5rem 1.5rem 3rem 1.5rem !important;
        max-width: 100% !important; position: relative; z-index: 1;
    }
    .sec {
        background: rgba(80,10,10,0.5); border-left: 4px solid #FF6B6B;
        padding: 0.45rem 0.9rem; border-radius: 5px; font-weight: bold;
        color: #FFB3B3; margin: 1rem 0 0.6rem 0;
        font-size: clamp(0.8rem, 2.5vw, 0.95rem); backdrop-filter: blur(6px);
    }
    .box-info {
        background: rgba(255,107,107,0.08); border-left: 4px solid #FF6B6B;
        padding: 0.7rem 1rem; border-radius: 6px;
        color: #FFB3B3; margin: 0.5rem 0;
    }
    .box-green {
        background: rgba(76,175,80,0.1); border-left: 4px solid #4CAF50;
        padding: 0.7rem 1rem; border-radius: 6px;
        color: #A5D6A7; margin: 0.5rem 0;
    }
    .anime-card {
        background: linear-gradient(135deg,
            rgba(20,5,40,0.85) 0%, rgba(80,10,10,0.65) 100%);
        border: 1px solid rgba(255,100,100,0.3); border-radius: 12px;
        padding: 1.2rem; margin: 0.8rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5); backdrop-filter: blur(8px);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,100,100,0.2);
        border-radius: 10px; padding: 0.6rem 0.8rem; backdrop-filter: blur(4px);
    }
    [data-testid="stMetricLabel"] { color: #FFB3B3 !important; }
    [data-testid="stMetricValue"] { color: #ffffff  !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; overflow-x: auto; flex-wrap: nowrap;
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(80,10,10,0.5); border-radius: 6px; padding: 8px 20px;
        font-weight: 600; color: #FFB3B3; white-space: nowrap;
        border: 1px solid rgba(255,100,100,0.2); backdrop-filter: blur(4px);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(160,30,30,0.85) !important;
        color: white !important;
        border-color: rgba(255,100,100,0.5) !important;
    }
    label { color: #FFB3B3 !important; }
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6B1A1A 0%, #9B2525 100%) !important;
        color: white !important;
        border: 1px solid rgba(255,100,100,0.4) !important;
        border-radius: 8px !important; font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Banner ─────────────────────────────────────────────────────────────────────
try:
    st.image("banner.png")
except Exception:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(10,25,60,0.96),
    rgba(60,10,10,0.9));padding:1.8rem;border-radius:14px;
    text-align:center;margin-bottom:1rem;
    border:1px solid rgba(255,100,100,0.2);">
        <h1 style="color:white;margin:0 0 0.3rem 0;">🎌 Anime Recommendation System</h1>
        <p style="color:#FF9999;font-weight:700;margin:0 0 0.1rem 0;">
            Applied Data Science, ML &amp; AI</p>
        <p style="color:#BDD7EE;margin:0;font-size:0.88rem;">
            E&amp;ICT Academy, IIT Guwahati &nbsp;·&nbsp;
            Content-Based Filtering &nbsp;·&nbsp;
            TF-IDF &nbsp;·&nbsp; Cosine Similarity</p>
    </div>""", unsafe_allow_html=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0d14',
    'axes.facecolor':   '#0d1020',
    'axes.edgecolor':   '#6B1A1A',
    'axes.labelcolor':  '#FFB3B3',
    'xtick.color':      '#FF9999',
    'ytick.color':      '#FF9999',
    'text.color':       '#FFB3B3',
    'grid.color':       '#1a0808',
    'grid.alpha':       0.5,
})

# ── Load & Build ───────────────────────────────────────────────────────────────
@st.cache_data
def load_and_build():
    anm = pd.read_csv("anime.csv")
    anm = anm.rename(columns={'rating': 'avg_rating'})
    anm['episodes'] = pd.to_numeric(anm['episodes'], errors='coerce')

    df = anm.copy()
    df['genre']        = df['genre'].fillna('')
    df['type']         = df['type'].fillna(df['type'].mode()[0])
    df['avg_rating']   = df['avg_rating'].fillna(df['avg_rating'].median())
    df['episodes']     = df['episodes'].fillna(df['episodes'].median())
    df['members_log']  = np.log1p(df['members'])
    df['episodes_log'] = np.log1p(df['episodes'])
    df = df.reset_index(drop=True)

    df['genre_clean'] = df['genre'].str.replace(',', ' ').str.strip()
    tfidf     = TfidfVectorizer(stop_words='english')
    tfidf_mat = tfidf.fit_transform(df['genre_clean'])

    ohe      = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    type_enc = ohe.fit_transform(df[['type']])

    scaler  = MinMaxScaler()
    num_mat = csr_matrix(scaler.fit_transform(
        df[['avg_rating', 'members_log', 'episodes_log']]
    ))

    # Sparse combined matrix → dense float32 (~3MB, memory safe)
    combined     = hstack([tfidf_mat * 3, type_enc * 2, num_mat * 1])
    combined_arr = combined.toarray().astype(np.float32)
    indices      = pd.Series(df.index, index=df['name']).drop_duplicates()

    return anm, df, combined_arr, indices

try:
    with st.spinner("🎌 Summoning the Anime Recommender..."):
        anm, anm_cb, combined_arr, indices = load_and_build()
    st.markdown(f"""
    <div class="box-green">
    ✅ <b>Ready!</b> — {len(anm):,} anime ·
    TF-IDF (genre ×3) + OneHot (type ×2) + MinMax (rating/members/episodes ×1) ·
    Feature matrix: {combined_arr.shape[0]:,} × {combined_arr.shape[1]}
    </div>""", unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ── Recommend (on-demand — no 1.2GB matrix) ────────────────────────────────────
def recommend(title, n=10):
    if title not in indices:
        return None
    idx       = indices[title]
    query_vec = combined_arr[idx:idx+1]
    sims      = cosine_similarity(query_vec, combined_arr).flatten()
    top_idx   = [i for i in np.argsort(sims)[::-1] if i != idx][:n]
    sim_vals  = [round(float(sims[i]), 4) for i in top_idx]
    result    = anm_cb.iloc[top_idx][
        ['name','genre','type','avg_rating','members','episodes']
    ].copy()
    result['similarity'] = sim_vals
    result = result.reset_index(drop=True)
    result.index += 1
    return result

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Data Overview", "🎌 Recommender"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">📋 Dataset Summary</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Anime",   f"{anm.shape[0]:,}")
    c2.metric("Features",      anm.shape[1])
    c3.metric("Unique Genres", anm['genre'].dropna()
              .str.split(',').explode().str.strip().nunique())
    c4.metric("Anime Types",   anm['type'].nunique())

    st.markdown("**First 10 rows**")
    st.dataframe(anm.head(10))

    st.markdown("**Descriptive Statistics**")
    st.dataframe(anm.describe().round(3))

    st.markdown('<div class="sec">⚠️ Missing Values</div>',
                unsafe_allow_html=True)
    miss = anm.isnull().sum().reset_index()
    miss.columns = ['Column', 'Missing Count']
    miss['Missing %'] = (miss['Missing Count'] / len(anm) * 100).round(2)
    st.dataframe(miss[miss['Missing Count'] > 0])

    st.markdown('<div class="sec">🤖 How It Works</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="box-info">
    <b>Features:</b> Genre (TF-IDF ×3) + Type (OneHot ×2) +
    Avg Rating / Members / Episodes (MinMax ×1)<br>
    <b>Similarity:</b> Cosine Similarity computed on-demand per request<br>
    <b>Memory safe:</b> ~3MB feature matrix instead of 1.2GB similarity matrix
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">🎌 Find Similar Anime</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        all_anime = sorted(anm_cb['name'].dropna().unique().tolist())
        default   = all_anime.index('Naruto') if 'Naruto' in all_anime else 0
        selected  = st.selectbox("🔍 Search & Select an Anime",
                                  options=all_anime, index=default)
    with c2:
        n_recs = st.selectbox("📋 Count",
                               options=[5, 10, 15, 20], index=1)

    if selected:
        sel = anm_cb[anm_cb['name'] == selected].iloc[0]
        eps = int(sel['episodes']) if not pd.isna(sel['episodes']) else 'N/A'

        st.markdown(f"""
        <div class="anime-card">
            <h3 style="color:#FF6B6B;margin:0 0 0.5rem 0;">🎬 {selected}</h3>
            <p style="color:#FFB3B3;margin:0.2rem 0;">
                <b style="color:#FF9999;">Genre:</b>
                {sel['genre'] or 'N/A'}
            </p>
            <p style="color:#FFB3B3;margin:0.2rem 0;">
                <b style="color:#FF9999;">Type:</b> {sel['type']}
                &nbsp;·&nbsp;
                <b style="color:#FF9999;">Episodes:</b> {eps}
                &nbsp;·&nbsp;
                <b style="color:#FF9999;">Rating:</b>
                {sel['avg_rating']:.2f} ⭐
                &nbsp;·&nbsp;
                <b style="color:#FF9999;">Members:</b>
                {int(sel['members']):,}
            </p>
        </div>""", unsafe_allow_html=True)

        if st.button("🎌 Get Recommendations", use_container_width=True):
            with st.spinner("Finding similar anime..."):
                recs = recommend(selected, n=n_recs)

            if recs is not None:
                st.markdown(
                    f'<div class="sec">✅ Top {n_recs} Similar to '
                    f'"{selected}"</div>', unsafe_allow_html=True)
                st.dataframe(recs)

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.barh(recs['name'], recs['similarity'],
                            color=sns.color_palette('YlOrRd', n_recs)[::-1],
                            edgecolor='#0a0d14')
                    ax.set_title('Cosine Similarity Score',
                                 fontweight='bold', fontsize=10)
                    ax.set_xlabel('Similarity')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.2, axis='x')
                    for p, v in zip(ax.patches, recs['similarity']):
                        ax.text(p.get_width()+0.003,
                                p.get_y()+p.get_height()/2,
                                f'{v:.4f}', va='center',
                                fontsize=8, color='#FFB3B3')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with c2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.barh(recs['name'], recs['avg_rating'],
                            color=sns.color_palette('Blues_r', n_recs),
                            edgecolor='#0a0d14')
                    ax.set_title('Average Rating',
                                 fontweight='bold', fontsize=10)
                    ax.set_xlabel('Rating')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.2, axis='x')
                    for p, v in zip(ax.patches, recs['avg_rating']):
                        ax.text(p.get_width()+0.05,
                                p.get_y()+p.get_height()/2,
                                f'{v:.2f}', va='center',
                                fontsize=8, color='#BDD7EE')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Genre breakdown
                st.markdown(
                    '<div class="sec">🎭 Genre Distribution</div>',
                    unsafe_allow_html=True)
                rec_genres = (recs['genre'].dropna()
                              .str.split(',').explode().str.strip())
                gdf = (pd.DataFrame(Counter(rec_genres).items(),
                                    columns=['Genre','Count'])
                       .sort_values('Count', ascending=False))
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.bar(gdf['Genre'], gdf['Count'],
                       color=sns.color_palette('YlOrRd', len(gdf))[::-1],
                       edgecolor='#0a0d14')
                ax.set_title(
                    f'Genres in Recommendations for "{selected}"',
                    fontweight='bold', fontsize=10)
                ax.set_xlabel('Genre'); ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.2, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("""
                <div class="box-green">
                ✅ <b>Recommendations complete!</b>
                </div>""", unsafe_allow_html=True)

    # Quick Try
    st.markdown('<div class="sec">⚡ Quick Try</div>',
                unsafe_allow_html=True)
    st.caption("Click any button to instantly see recommendations")

    quick_list = ['Naruto','Death Note','One Piece',
                  'Fullmetal Alchemist: Brotherhood',
                  'Attack on Titan','Dragon Ball Z']
    cols = st.columns(3)
    for i, name in enumerate(quick_list):
        with cols[i % 3]:
            if st.button(f"🎌 {name}", use_container_width=True,
                         key=f"q_{i}"):
                with st.spinner(f"Finding similar to {name}..."):
                    recs = recommend(name, n=10)
                if recs is not None:
                    st.markdown(f"**Top 10 for {name}:**")
                    st.dataframe(
                        recs[['name','genre','type',
                              'avg_rating','similarity']])