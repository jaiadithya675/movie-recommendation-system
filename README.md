# 🎬 Movie Recommendation System

![CI](https://github.com/YOUR-USERNAME/movie-recommender/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end recommender system built on the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset. It implements and compares **six models** (baselines, content-based, collaborative filtering, matrix factorization and a hybrid) with a **leakage-free temporal evaluation**, unit tests, CI, a CLI, a Streamlit web app and Docker support.

## Highlights
- Clean, reusable package (`movierec/`) with one shared interface for every model
- Realistic evaluation: each user's *latest* 20% of ratings is held out (no future data leaks into training)
- Both **rating-prediction** (RMSE, MAE) and **top-N ranking** metrics (Precision, Recall, NDCG, Hit Rate, Coverage)
- Cold-start handling: unknown users fall back to popularity
- Automated tests (`pytest`), linting (`ruff`) and GitHub Actions CI
- Interactive demo (Streamlit) and CLI; Dockerfile included

## Models
| Model | Type | Idea |
|---|---|---|
| BayesianAverage | Baseline | Item mean rating shrunk toward the global mean (IMDb-style weighted rating) |
| MostPopular | Baseline | Rank by number of ratings |
| ContentBased | Content | TF-IDF on genres; user profile is built from the movies they rated above their own average |
| ItemKNN | Collaborative | Adjusted-cosine item similarity, co-rating shrinkage, top-40 neighbours, damped weighted average |
| SVD | Matrix factorization | Truncated SVD (50 factors) of the user-mean-centred rating matrix |
| Hybrid | Ensemble | Weighted blend (0.4 SVD, 0.4 ItemKNN, 0.2 Content) of min-max normalised scores |

## Quick start
```bash
git clone https://github.com/YOUR-USERNAME/movie-recommender.git
cd movie-recommender
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
make install            # or: pip install -r requirements-dev.txt
make data               # downloads MovieLens ml-latest-small (~1 MB)
make evaluate           # trains + evaluates all models, saves results/metrics.csv
make app                # Streamlit demo at http://localhost:8501
```
> If the dataset isn't downloaded, the code falls back to a small **synthetic** dataset (with a warning) so tests and demos still run. Report results only from real MovieLens data.

### CLI
```bash
python -m movierec.cli recommend --user 1 --model Hybrid --k 10
python -m movierec.cli similar --title "Toy Story" --k 10
python -m movierec.cli evaluate --k 10
```

### Docker
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

## Evaluation methodology
1. Items with fewer than 5 ratings are removed.
2. **Temporal per-user split:** the latest 20% of each user's ratings form the test set.
3. Models train only on the training set.
4. **RMSE/MAE** are computed on all test ratings (rating-scale models only).
5. **Ranking metrics @10:** a test movie rated >= 4.0 counts as relevant; models rank all movies the user has not seen in training. Coverage is the share of the catalogue that gets recommended to at least one user.

## Results
Run `make evaluate` and paste your table here (numbers below are placeholders):

| model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|---|---|---|
| BayesianAverage | _run to fill_ | | | | | | |
| MostPopular | | | | | | | |
| ContentBased | | | | | | | |
| ItemKNN | | | | | | | |
| SVD | | | | | | | |
| Hybrid | | | | | | | |

**Analysis (write after running):** which model wins on error vs ranking, how much the hybrid helps, and the accuracy-vs-coverage trade-off.

## Project structure
```
movie-recommender/
├── movierec/            # library: data.py, models.py, evaluate.py, cli.py
├── tests/               # pytest unit tests
├── scripts/             # dataset download
├── app.py               # Streamlit demo
├── .github/workflows/   # CI (ruff + pytest)
├── Dockerfile  Makefile  requirements*.txt  LICENSE
```

## Design decisions & limitations
- Genres are the only content feature; adding tags, tag-genome or plot embeddings would improve cold-start items.
- Item-item similarity is a dense matrix, which is fine for ~10k movies but needs approximate nearest neighbours at larger scale.
- Offline metrics are proxies; real quality needs online A/B tests.
- Popularity bias: MovieLens test data favours popular items, so check Coverage alongside accuracy.

## Future work
- Implicit-feedback ALS / BPR, neural collaborative filtering
- Hyperparameter search and cross-validation
- Diversity/novelty metrics, explanations ("because you watched ...")
- FastAPI service and model persistence

## License
MIT. Dataset © GroupLens Research; see the MovieLens terms of use.
