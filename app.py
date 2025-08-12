# app.py — E-commerce Recommender (Hybrid: ALS + Content + MMR) — Light theme default
import os
import numpy as np
import pandas as pd
import streamlit as st
from html import escape as _esc

from rec_core.data import load_csv, normalize_items, normalize_ratings, filter_sparsity, valid_url
from rec_core.models import (
    build_content_model, content_recommend,
    build_als_model, als_recommend, get_popular_items,
    finalize_rank
)
from rec_core.metrics import evaluate_models
from rec_core.explain import content_explanations, als_because

st.set_page_config(page_title="E-commerce Recommender — Hybrid", page_icon="🛍️", layout="wide")

THEMES = {
    "Light": {
        "bg": "#F7F8FA","surface": "#FFFFFF","text": "#0F172A","muted": "#5B6472",
        "primary": "#3B82F6","accent": "#8B5CF6","shadow": "rgba(2, 6, 23, 0.08)",
        "card_border": "rgba(15, 23, 42, 0.06)","skeleton": "#E5E7EB",
    },
    "Dark": {
        "bg": "#0B1220","surface": "#0F172A","text": "#E5E7EB","muted": "#98A2B3",
        "primary": "#60A5FA","accent": "#A78BFA","shadow": "rgba(2, 6, 23, 0.45)",
        "card_border": "rgba(226, 232, 240, 0.08)","skeleton": "#1F2937",
    },
}
def inject_theme_css(theme_name: str):
    p = THEMES.get(theme_name, THEMES["Light"])
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{ background: {p["bg"]} !important; }}
    * {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }}
    h1,h2,h3,h4,h5,h6, .stMarkdown p, .stMarkdown span {{ color: {p["text"]} !important; }}
    .app-hero {{ display:flex; align-items:center; justify-content:space-between;
      padding: 12px 18px; background: {p["surface"]}; border-radius: 16px;
      border: 1px solid {p["card_border"]}; box-shadow: 0 8px 30px {p["shadow"]};
      margin-bottom: 18px; }}
    .title {{ font-size: 20px; font-weight: 700; letter-spacing: 0.2px; }}
    .pill {{ display:inline-block; margin-left: 8px; padding: 3px 10px; border-radius: 999px;
      background: linear-gradient(90deg, {p["primary"]}, {p["accent"]}); color: white; font-weight: 600; font-size: 12px; }}
    .subtle {{ color: {p["muted"]}; font-size: 13px; }}
    .m-card {{ background: {p["surface"]}; border: 1px solid {p["card_border"]}; border-radius: 14px;
      padding: 12px; box-shadow: 0 8px 30px {p["shadow"]}; overflow:hidden; }}
    .m-card img {{ width: 100%; height: 160px; object-fit: cover; border-radius: 10px; background: {p["skeleton"]}; margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; width: 100%; }}
    .rec-title {{ font-weight: 600; margin: 6px 0; line-height:1.25 }}
    .meta {{ color: {p["muted"]}; font-size: 12px; margin-bottom:4px }}
    .cta {{ display:inline-block; padding:8px 12px; border-radius:10px; margin-top:8px; margin-right:8px;
      background:{p["primary"]}; color:white; text-decoration:none; font-weight:600; }}
    .cta.secondary {{ background: transparent; color: {p["primary"]}; border: 1px solid {p["primary"]}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ Controls")
    theme_choice = st.selectbox("Theme", ["Light", "Dark"], index=0)
    inject_theme_css(theme_choice)

st.markdown("""
<div class="app-hero">
  <div class="title">🛍️ E-commerce Recommendation System <span class="pill">Hybrid</span></div>
  <div class="subtle">ALS × Content · Popularity fallback · MMR · Evaluation</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📥 Data")
    r_file = st.file_uploader("Upload ratings CSV", type=["csv"], key="ratings_up")
    i_file = st.file_uploader("Upload items CSV", type=["csv"], key="items_up")
    st.caption("ratings[user_id,item_id,rating,timestamp], items[item_id,title,category,description,price,image_url,product_url*]")

default_r = os.path.join("sample_data","ratings.csv")
default_i = os.path.join("sample_data","items.csv")
ratings_raw = load_csv(r_file, default_path=default_r)
items_raw   = load_csv(i_file, default_path=default_i)

if ratings_raw is None or items_raw is None:
    st.info("Upload CSVs in the sidebar or keep sample_data/* as default.")
    st.stop()

ratings = normalize_ratings(ratings_raw)
items   = normalize_items(items_raw)

pop_by_item = ratings.groupby("item_id")["rating"].sum()
title_map = (items.assign(pop=items["item_id"].map(pop_by_item).fillna(0))
             .sort_values(["pop","title"], ascending=[False, True]))

if "cart" not in st.session_state:
    st.session_state.cart = []

@st.cache_resource(show_spinner=True)
def _content_model_cached(items_df):
    return build_content_model(items_df)

@st.cache_resource(show_spinner=True)
def _als_model_cached(ratings_df, **kwargs):
    return build_als_model(ratings_df, filter_sparsity_fn=filter_sparsity, **kwargs)

with st.sidebar:
    st.markdown("### ⚙️ Modeling")
    K = st.slider("Top-K", 5, 50, 10, 1)
    w_als = st.slider("Hybrid weight (ALS ↔ Content)", 0.0, 1.0, 0.6, 0.05)
    st.caption("0 = content only · 1 = ALS only")
    st.markdown("**ALS**")
    factors = st.slider("Factors", 10, 200, 70, 5)
    reg = st.slider("Regularization", 0.001, 0.2, 0.05, 0.001)
    iters = st.slider("Iterations", 5, 50, 20, 1)
    alpha = st.slider("Alpha (confidence)", 1.0, 50.0, 15.0, 1.0)
    min_user = st.slider("Min interactions per user", 1, 10, 2, 1)
    min_item = st.slider("Min interactions per item", 1, 10, 2, 1)
    st.markdown("**Diversity (MMR)**")
    use_mmr = st.toggle("Enable MMR re-ranking", value=True)
    lambda_div = st.slider("MMR λ (more → more diversity)", 0.0, 0.9, 0.35, 0.05)
    pool_size = st.slider("MMR candidate pool", 20, 300, 120, 10)
    st.markdown("**Card renderer**")
    renderer = st.selectbox("Result cards", ["Streamlit columns (no HTML)", "HTML cards"], index=0)

content_model = _content_model_cached(items)
als_pack = _als_model_cached(ratings, factors=factors, reg=reg, iters=iters, alpha=alpha, min_user=min_user, min_item=min_item)

def show_cards_html(item_ids, items_df, seed_id=None, content_model=None, user_hist_titles=None):
    df = items_df.set_index("item_id").reindex(item_ids).reset_index()
    explain = {}
    if seed_id and content_model:
        explain = content_explanations(seed_id, item_ids, content_model, topn=3)
    because = als_because(user_hist_titles or [], max_items=3) if user_hist_titles else []
    cards = []
    for _, r in df.iterrows():
        iid = r.get("item_id")
        title = _esc((r.get("title") or str(iid))[:140])
        cat = _esc((r.get("category") or "Uncategorized"))
        price = r.get("price")
        price_str = f"${float(price):,.2f}" if isinstance(price, (int, float)) and not pd.isna(price) else "—"
        img = r.get("image_url", "")
        img_tag = f"<img src='{_esc(img)}' alt='product image' />" if valid_url(img) else f"<div style='width:100%;height:160px;border-radius:10px;background:rgba(0,0,0,.08);margin-bottom:8px'></div>"
        link = r.get("product_url", "")
        link = link if valid_url(link) else "#"
        because_txt = ""
        if seed_id and iid in explain and explain[iid]:
            because_txt = f"<div class='meta'>Similar terms: {', '.join(_esc(t) for t in explain[iid])}</div>"
        elif because:
            because_txt = f"<div class='meta'>Because you viewed: {', '.join(_esc(t) for t in because)}</div>"
        cards.append(
            "<div class='m-card'>"
            f"{img_tag}"
            f"<div class='rec-title'>{title}</div>"
            f"<div class='meta'>{cat}</div>"
            f"<div class='meta'>Price: {price_str}</div>"
            f"{because_txt}"
            f"<a class='cta' href='{_esc(link)}' target='_blank' rel='noopener'>Open</a>"
            f"<a class='cta secondary' href='javascript:void(0);' onclick=\"navigator.clipboard.writeText('{_esc(link if link != '#' else title)}');\">Copy link</a>"
            "</div>"
        )
    st.markdown("<div class='grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

def show_cards_streamlit(item_ids, items_df, seed_id=None, content_model=None, user_hist_titles=None, cols_per_row=3):
    df = items_df.set_index("item_id").reindex(item_ids).reset_index()
    explain = {}
    if seed_id and content_model:
        explain = content_explanations(seed_id, item_ids, content_model, topn=3)
    because = als_because(user_hist_titles or [], max_items=3) if user_hist_titles else []
    chunks = [df[i:i+cols_per_row] for i in range(0, len(df), cols_per_row)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, (row_idx, r) in zip(cols, chunk.iterrows()):
            key_token = f"{r['item_id']}_{row_idx}"
            with col:
                try:
                    ctx = st.container(border=True)
                except TypeError:
                    ctx = st.container()
                with ctx:
                    img = r.get("image_url", "")
                    if valid_url(img):
                        st.image(img, use_container_width=True)
                    title = (r.get("title") or str(r.get("item_id", "")))[:140]
                    st.markdown(f"**{title}**")
                    cat = (r.get("category") or "Uncategorized")
                    st.caption(cat)
                    price = r.get("price")
                    price_str = f"${float(price):,.2f}" if isinstance(price, (int, float)) and not pd.isna(price) else "—"
                    st.write(f"Price: {price_str}")
                    if seed_id and r["item_id"] in explain and explain[r["item_id"]]:
                        st.caption("Similar terms: " + ", ".join(explain[r["item_id"]]))
                    elif because:
                        st.caption("Because you viewed: " + ", ".join(because))
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("Add to cart", key=f"cart_{key_token}"):
                            if "cart" not in st.session_state:
                                st.session_state.cart = []
                            if r["item_id"] not in st.session_state.cart:
                                st.session_state.cart.append(r["item_id"])
                                st.toast("Added to cart 🛒", icon="✅")
                            else:
                                st.toast("Already in cart", icon="ℹ️")
                    link = r.get("product_url", "")
                    with c2:
                        if valid_url(link):
                            st.link_button("Open", link, use_container_width=True)
                        else:
                            st.button("Open", key=f"open_disabled_{key_token}", disabled=True)
                    with c3:
                        copy_target = link if valid_url(link) else title
                        if st.button("Copy", key=f"copy_{key_token}"):
                            st.toast("Copy the text below (Cmd/Ctrl+C)", icon="📋")
                            st.code(copy_target)

def render_cards(ids, items_df, renderer: str, seed_id=None, content_model=None, user_hist_titles=None):
    if renderer.startswith("Streamlit"):
        show_cards_streamlit(ids, items_df, seed_id=seed_id, content_model=content_model, user_hist_titles=user_hist_titles)
    else:
        show_cards_html(ids, items_df, seed_id=seed_id, content_model=content_model, user_hist_titles=user_hist_titles)

tab_rec, tab_eval, tab_cart = st.tabs(["🔎 Recommend", "🧪 Evaluate", "🛒 Cart"])

with tab_rec:
    st.markdown("#### What do you want recommendations **based on**?")
    mode = st.radio("", ["A product (content)", "A user (ALS/hybrid)"], horizontal=True)
    with st.sidebar:
        with st.expander(f"🛒 Cart ({len(st.session_state.cart)})", expanded=False):
            if st.session_state.cart:
                cart_df = items[items["item_id"].isin(st.session_state.cart)][["title","price"]]
                st.table(cart_df.reset_index(drop=True))
                total = cart_df["price"].dropna().sum()
                st.write(f"**Subtotal:** ${total:,.2f}" if not pd.isna(total) else "**Subtotal:** —")
                if st.button("Clear cart"):
                    st.session_state.cart = []
                    st.experimental_rerun()
            else:
                st.caption("Your cart is empty.")
    if mode == "A product (content)":
        titles = title_map["title"].tolist()
        pick = st.selectbox("Pick a product", titles)
        item_id = title_map[title_map["title"] == pick]["item_id"].iloc[0]
        cand_k = max(120, K)
        content_ids = content_recommend(item_id, content_model, top_k=cand_k)
        final_ids = finalize_rank(content_ids, [], K=K, w_als=0.0, use_mmr=True,
                                  lambda_div=0.35, content_model=content_model, pool_size=120)
        st.markdown("##### Recommendations")
        render_cards(final_ids, items, "Streamlit columns (no HTML)", seed_id=item_id, content_model=content_model)
    else:
        users = ratings["user_id"].unique().tolist()
        user_id = st.selectbox("Pick a user", users)
        user_hist = ratings[ratings["user_id"] == user_id].sort_values("timestamp")
        cand_k = max(120, K)
        content_ids = []
        user_hist_titles = None
        if len(user_hist) > 0:
            last_item = user_hist["item_id"].iloc[-1]
            content_ids = content_recommend(last_item, content_model, top_k=cand_k)
            user_hist_titles = items.set_index("item_id").loc[user_hist["item_id"].tolist()]["title"].fillna("").tolist()
        als_ids = als_recommend(user_id, als_pack, top_k=cand_k) if als_pack is not None else []
        if (not als_ids) and (not content_ids):
            st.info("New or sparse user — showing popular products.")
            items_pop = get_popular_items(ratings, items, top_n=K)
            render_cards(items_pop, items, "Streamlit columns (no HTML)", user_hist_titles=user_hist_titles, content_model=content_model)
        else:
            final_ids = finalize_rank(content_ids, als_ids, K=K, w_als=0.6, use_mmr=True,
                                      lambda_div=0.35, content_model=content_model, pool_size=120)
            st.markdown("##### Recommendations")
            render_cards(final_ids, items, "Streamlit columns (no HTML)", seed_id=(last_item if len(user_hist)>0 else None),
                         content_model=content_model, user_hist_titles=user_hist_titles)
            st.balloons()

with tab_eval:
    st.markdown("### 📊 Evaluate models (Recall/MAP/NDCG + Coverage/Diversity/Novelty)")
    min_inter_per_user = st.slider("Only evaluate users with ≥N interactions", 2, 10, 3, 1)
    if st.button("Run quick evaluation", type="primary", use_container_width=True):
        with st.spinner("Evaluating models…"):
            metrics = evaluate_models(
                ratings, items, content_model, als_pack, K=K, min_interactions=min_inter_per_user,
                use_mmr=True, lambda_div=0.35, w_als=0.6, pool_size=120
            )
        st.dataframe(metrics, use_container_width=True)
        st.caption("ALS uses user history; Content uses last seen item; MMR reduces near-duplicates.")

with tab_cart:
    st.markdown("### 🛒 Cart")
    if st.session_state.cart:
        cart_df = items[items["item_id"].isin(st.session_state.cart)].copy()
        cart_df = cart_df[["item_id","title","price","product_url"]]
        cart_df["price"] = cart_df["price"].apply(lambda x: f"${x:,.2f}" if isinstance(x,(int,float)) and not pd.isna(x) else "—")
        st.table(cart_df.reset_index(drop=True))
        raw_total = items[items["item_id"].isin(st.session_state.cart)]["price"].dropna().sum()
        st.write(f"**Subtotal:** ${raw_total:,.2f}" if not pd.isna(raw_total) else "**Subtotal:** —")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear cart (all)"):
                st.session_state.cart = []
                st.experimental_rerun()
        with c2:
            st.button("Checkout (demo)", disabled=True)
    else:
        st.info("Your cart is empty. Add some items from the recommendations tab.")
