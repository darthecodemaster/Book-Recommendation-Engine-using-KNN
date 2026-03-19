# Book Recommendation Engine using K-Nearest Neighbors

> **FreeCodeCamp Machine Learning with Python Certification — Project 4**

A collaborative filtering recommendation system that suggests similar books based on user rating patterns, built using the K-Nearest Neighbors algorithm and the Book-Crossings dataset.

---

## 🎯 Project Overview

This project implements a **memory-based collaborative filtering** system. Instead of analyzing book content (title, genre, author), it finds books that tend to be rated similarly by the same users — the core idea behind platforms like Amazon's "Customers who bought this also bought..." feature.

**Algorithm:** K-Nearest Neighbors with cosine similarity  
**Dataset:** Book-Crossings (1.1M ratings · 270K books · 90K users)  
**Platform:** Google Colaboratory / Jupyter Notebook

---

## 🚀 Quick Start

### Option A: Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook** and upload `book_recommendation_knn.ipynb`
3. Click **Runtime → Run all**
4. Share the link with **"Anyone with the link"** enabled

### Option B: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/book-recommendation-knn.git
cd book-recommendation-knn

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook book_recommendation_knn.ipynb
```

---

## 📁 Project Structure

```
book-recommendation-knn/
│
├── book_recommendation_knn.ipynb   # Main notebook (solution)
├── README.md                       # This file
├── JOURNAL.md                      # Development journal & notes
├── requirements.txt                # Python dependencies
└── rating_distributions.png        # Optional visualization output
```

---

## 📊 Dataset

The **Book-Crossings** dataset was collected by Cai-Nicolas Ziegler from the Book-Crossing community.

| File | Records | Description |
|------|---------|-------------|
| BX-Books.csv | 271,379 | Book metadata (title, author, year) |
| BX-Users.csv | 278,858 | User demographics |
| BX-Book-Ratings.csv | 1,149,780 | Explicit (1–10) and implicit (0) ratings |

> The dataset is automatically downloaded in the notebook — no manual setup needed.

---

## 🔧 How It Works

### Step 1 — Data Filtering

Most users rate very few books. To ensure statistical significance:
- Remove users with **fewer than 200 ratings**
- Remove books with **fewer than 100 ratings**

This drastically reduces noise and improves recommendation quality.

### Step 2 — Build the Book-User Matrix

```
         User_A  User_B  User_C  ...
Book_1      8       0       5
Book_2      0       7       0
Book_3      6       9       0
```

Each row is a book's "rating fingerprint" across all users.

### Step 3 — Train KNN with Cosine Similarity

Cosine similarity measures the *angle* between two rating vectors:
- **Score = 0** → identical rating patterns (perfect match)
- **Score = 1** → completely opposite patterns (no similarity)

```python
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
model.fit(book_sparse_matrix)
```

### Step 4 — Query for Recommendations

```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```

Returns:
```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.7939],
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448],
    ['Interview with the Vampire', 0.7345],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178]
  ]
]
```
## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
matplotlib>=3.4.0
requests>=2.25.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🧠 Key Concepts

| Concept | Explanation |
|---------|-------------|
| Collaborative Filtering | Recommends based on *user behavior*, not item content |
| Cosine Similarity | Measures angle between vectors — ignores rating scale differences |
| Sparse Matrix | Efficient storage for mostly-empty user-book grids |
| KNN | Finds K most similar items by distance metric |

---

## 📈 Results

After filtering, the working dataset contains:
- ~800–1,000 books (those with ≥100 ratings)
- ~400–500 active users (those with ≥200 ratings)

The model successfully identifies semantically related books — e.g., Anne Rice vampire novels cluster together, Harry Potter books are grouped, etc.

---

## 🔮 Future Improvements

- **Matrix Factorization (SVD)** — better for sparse data, learns latent factors
- **Content-Based Hybrid** — combine KNN with genre/author features
- **Dynamic Thresholds** — tune filtering thresholds with cross-validation
- **Flask API** — wrap `get_recommends()` in a REST endpoint

---

## 📄 License

This project is for educational purposes as part of the FreeCodeCamp Machine Learning certification.

---

## 🙏 Credits

- **Dataset:** Book-Crossing Dataset by Cai-Nicolas Ziegler
- **Project:** [FreeCodeCamp Machine Learning with Python Certification](https://www.freecodecamp.org/learn/machine-learning-with-python/)
