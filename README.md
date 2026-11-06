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
make all                # lint + tests + evaluation + plots + CLI demos -> results/
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
All numbers below were produced by `make all` on the real **MovieLens ml-latest-small** dataset (raw: 610 users, 9,724 rated movies, 100,836 ratings, 98.3% sparse) and are saved in `results/`. After removing movies with fewer than 5 ratings, 3,650 movies and 90,274 ratings remain (95.95% sparse); 72,434 ratings are used for training and the latest 17,840 (20% of each user's history) for testing.

| model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|---|---|---|
| BayesianAverage | 0.9731 | 0.7525 | 0.0417 | 0.0328 | 0.0549 | 0.2627 | 0.013 |
| MostPopular | n/a | n/a | 0.0571 | 0.0510 | 0.0755 | 0.3169 | 0.026 |
| ContentBased | n/a | n/a | 0.0098 | 0.0061 | 0.0102 | 0.0898 | 0.604 |
| ItemKNN | **0.8985** | **0.6811** | 0.0698 | 0.0662 | 0.0949 | 0.3746 | 0.236 |
| SVD | 0.9443 | 0.7278 | 0.0673 | 0.0663 | 0.0893 | **0.4068** | 0.160 |
| **Hybrid** | n/a | n/a | **0.0727** | **0.0734** | **0.1007** | 0.3864 | 0.237 |

![Model comparison](results/model_comparison.png)
![Dataset overview](results/eda_overview.png)

### Analysis
- **Personalisation beats popularity.** The Hybrid's NDCG@10 (0.1007) is 33% higher than MostPopular (0.0755), and it reaches about 9x the catalogue coverage (23.7% vs 2.6%), so it recommends far more than the same blockbusters.
- **ItemKNN is the most accurate rating predictor** (RMSE 0.8985 vs 0.9731 for the baseline, 7.7% lower). **SVD has the best hit rate**: 40.7% of users with held-out favourites get at least one in their top 10 in their top 10.
- **Blending helps.** The Hybrid beats its best single component (ItemKNN) on NDCG by about 6% and on recall by 11%, because the models make different mistakes.
- **Genre-only content filtering is weak** (NDCG 0.0102) but has the highest coverage (60%). Genres are a coarse signal: user 1 rated 28 animated films at 4.68 on average (their overall mean is 4.35), so the model correctly learns they like animation and then suggests mostly Disney titles, ignoring everything that makes a film good. Adding tags, plot embeddings or popularity priors would address this.
- **Absolute precision looks low (about 7%)** because the task is hard: each user has only a handful of relevant held-out movies among thousands of candidates, so values like these are typical for MovieLens-style temporal splits.

Caveats: single split with one random seed, no confidence intervals, and default hyperparameters (no tuning). Offline metrics are proxies for real user satisfaction.

## Execution output
`make all` (about 25 seconds on a laptop-class CPU) runs lint, tests, evaluation, report generation and both CLI demos, and writes everything to `results/execution_log.txt`. Excerpt:
```text
$ ruff check .                      All checks passed!
$ python -m pytest -v               6 passed
$ python -m movierec.cli evaluate   (table above)
$ python -m movierec.cli similar --title "Toy Story" --k 5
Because you picked: Toy Story (1995)
 Toy Story 2 (1999) | Aladdin (1992) | Toy Story 3 (2010)
 Wallace & Gromit: The Wrong Trousers (1993) | Back to the Future (1985)
```
`results/sample_recommendations.txt` shows each model's top 10 for user 1 next to the movies that user *actually* rated highly later, plus "movies similar to ..." lists (for example, The Dark Knight leads to The Dark Knight Rises and Batman Begins).

| File in `results/` | Content |
|---|---|
| `metrics.csv` | Final metrics table |
| `dataset_summary.json` | Users, movies, ratings, sparsity, split sizes |
| `model_comparison.png`, `eda_overview.png` | Charts |
| `sample_recommendations.txt` | Example recommendations and similar-movie lists |
| `execution_log.txt` | Full log of lint, tests, evaluation and CLI runs |

## Project structure
```
movie-recommender/
├── movierec/            # library: data.py, models.py, evaluate.py, cli.py
├── tests/               # pytest unit tests
├── scripts/             # download_data.py, make_report.py, run_all.sh
├── results/             # metrics, charts, sample recommendations, execution log
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
