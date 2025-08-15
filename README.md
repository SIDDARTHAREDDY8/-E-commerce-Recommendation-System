# 🛍️ E-commerce Recommendation System — Hybrid (ALS × Content)
Tech: Python · Streamlit · pandas · scikit-learn · SciPy · implicit (ALS)
A production-style recommender that blends Collaborative Filtering (ALS) with Content-based similarity, adds MMR diversity, and falls back to popularity for cold-start. Includes an elegant UI (light/dark), “add to cart”, and an evaluation tab (Recall/MAP/NDCG + Coverage/Diversity/Novelty).

🚀 **Live Demo:** [Click here to view the app](https://chbazwgidrxkwkxkno7dmj.streamlit.app)  
📂 **GitHub Repo:** [crypto-dashboard]([https://github.com/SIDDARTHAREDDY8/](https://github.com/SIDDARTHAREDDY8/-E-commerce-Recommendation-System))  
💼 **LinkedIn:** [Siddartha Reddy Chinthala](https://www.linkedin.com/in/siddarthareddy9)

---

## ✨ Features

- **Hybrid ranking**: ALS (implicit matrix factorization) + TF-IDF/KNN content similarity
- **MMR re-ranking**: increases diversity by penalizing near-duplicates
- **Cold-start**: popularity fallback for new users/sparse histories
- **Beautiful UI**: light/dark theme, image cards (graceful placeholder), “Open”, “Copy”, “Add to cart”
- **Evaluation**: one-click offline metrics (Recall@K, MAP@K, NDCG@K, Coverage, Diversity, Novelty)
- **Robust loaders**: flexible CSV normalization (e.g., asin → item_id, order_date → timestamp)

---

## 📦 Project Structure

```bash
ecomm-recs/
  app.py                       # Streamlit UI & orchestration
  rec_core/
    __init__.py
    data.py                    # loaders, normalization, sparsity filters, URL checks
    models.py                  # TF-IDF+KNN, ALS (implicit), hybrid blend, MMR
    metrics.py                 # Recall/MAP/NDCG + Coverage/Diversity/Novelty
    explain.py                 # “because” terms (content) + recent history (ALS)
  sample_data/
    items.csv
    ratings.csv
  experiments/
    results.csv
    figures/
  notebooks/
    01_eda.ipynb
    02_modeling.ipynb
  requirements.txt
  packages.txt                 # system deps for Streamlit Cloud (OpenBLAS/OpenMP)
  runtime.txt                  # python-3.11 (for cloud runtime)
  README.md
  MODEL_CARD.md
```

---

## 🔧 Quick Start (Local)

```bash
# 1) (Optional) create & activate virtual env
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

---

## ☁️ Deploy (Streamlit Community Cloud)
1. This repo already contains the right files (requirements.txt, packages.txt, runtime.txt).
2. Push to GitHub (branch main).
3. Go to https://share.streamlit.io → New app → select repo, branch main, main file app.py.
4. In Manage app → Settings → Advanced, set Python version = 3.11.
5. **Add env vars**:
```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

---

**packages.txt**
```bash
libgomp1
libopenblas0
```

---

## 📊 Data Schemas

**Provide two CSVs (or start with sample_data/).**
1. **ratings.csv** (implicit interactions)
   ```bash
 | column      | required | notes                                       |
| ----------- | -------- | ------------------------------------------- |
| `user_id`   | ✅        | user identifier                             |
| `item_id`   | ✅        | product id (`asin` normalized)              |
| `rating`    | ➕        | implicit strength/quantity (default 1.0)    |
| `timestamp` | ➕        | ISO/date string (used for train/test split) |
  ```

2. **items.csv**
   | column        | required | notes                                      |
| ------------- | -------- | ------------------------------------------ |
| `item_id`     | ✅        | product id                                 |
| `title`       | ✅        | used in UI and TF-IDF                      |
| `category`    | ➕        | improves content similarity                |
| `description` | ➕        | improves content similarity                |
| `price`       | ➕        | numeric; string cleaned if symbols present |
| `image_url`   | ➕        | optional; placeholder if missing/invalid   |
| `product_url` | ➕        | optional; disables “Open” if missing       |

**Column normalization supported**:
item_id ⇢ asin, asin/isbn · rating ⇢ quantity · timestamp ⇢ order_date, time, reviews.date · price ⇢ purchase price per unit, list price per unit.

---

## 🧠 How It Works

1. **Content model**
TF-IDF (1–2 n-grams, English stopwords) on title + category + description; cosine KNN returns similar items.
2.**ALS model (implicit MF)**
Sparse user×item matrix from implicit feedback; trains implicit ALS; recommends unseen items (filters already-seen). Fallback to a fast item-item score if ALS isn’t available.
3. **Hybrid blend**
Combine content & ALS lists with weight w_als (sidebar slider).
4. **MMR diversity**
Re-rank with Maximal Marginal Relevance to reduce near-duplicates (λ slider).
5. **Cold-start**
Popularity fallback for new/sparse users; content similarity covers new items.

---
## 🧪 Evaluation (Offline)
- **Recall@K** – does top-K include the next item?
- **MAP@K** – rank-aware precision at first hit
- **NDCG@K** – higher if relevant items rank near the top
- **Coverage** – fraction of catalog recommended at least once
- **Diversity** – average (1 − cosine) among recommended items
- **Novelty** – −log(popularity) averaged over recommendations
- **Open the 🧪 Evaluate tab (set K & min interactions in the sidebar).**

---

## 🖥️ Using the App
- **🔎 Recommend**
  - **A product (content)**: pick a title, see similar items + “Similar terms: …”.
  - **A user (ALS/hybrid)**: pick a user, see blended recs + “Because you viewed: …”.
- **Cards**: image (or placeholder), price, Open, Copy, Add to cart.
- **🛒 Cart** – shows selected items & subtotal.
- **Theme** – Light/Dark toggle (default Light).
- **Sidebar** – tune K, ALS params (factors/reg/iters/alpha, min user/item), MMR λ, hybrid weight.

---

## 🧩 Design Choices
- **Implicit ALS** for sparse interaction signals
- **Content model** for explainability & new items
- **MMR** for catalog exploration (less duplication)
- **Popularity fallback** to avoid empty states
- **Modular core (rec_core/*)** for easy swaps (BERT embeddings, rerankers)

---

## 🧾 Model Card (Summary)
- **Data**: implicit interactions; product text; optional price/images/links
- **Assumptions**: more interactions ≈ stronger preference; text reflects similarity
- **Limitations**: popularity bias; text quality matters; ALS needs minimum interactions
- **Safety/Privacy**: user IDs only; no PII; avoid re-identification
See MODEL_CARD.md for full details.

---

## 🐛 Troubleshooting
- **“App in the oven” / hangs**: set Python 3.11 (Manage app → Settings), keep packages.txt with OpenBLAS/OpenMP.
- **implicit build errors**: ensure Python 3.11; try libopenblas0 ↔ libopenblas-dev.
- **Duplicate widget IDs**: fixed with unique keys in code.
- **No images/links**: placeholders & disabled buttons handled gracefully.

---

## 🧭 Roadmap
- Sequential next-item models (GRU4Rec/Transformers)
- Business-aware re-rank (margin/stock/SLA)
- Fairness/exposure & serendipity metrics
- Content embeddings (MiniLM/BERT)
- A/B harness + logging (impressions/clicks)

---

## 🙌 Acknowledgements
- ```bash implicit ```
- ```bash scikit-learn, numpy, scipy, pandas ```
- ```bash Streamlit ```

---

## 👤 Author

**[Siddartha Reddy Chinthala](https://www.linkedin.com/in/siddarthareddy9)**  
🎓 Master’s in CS | Aspiring Data Scientist  
🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/siddarthareddy9)

⭐️ Show Some Love
If you like this project, don’t forget to ⭐️ the repo and share it!


