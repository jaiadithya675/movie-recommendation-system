# 🎬 Movie Recommendation System

![CI](https://github.com/YOUR-USERNAME/movie-recommender/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end recommender system on the [MovieLens](https://grouplens.org/datasets/movielens/latest/) `ml-latest-small` dataset. It implements **six models** (two baselines, content-based, item-based collaborative filtering, matrix factorization and a hybrid), tunes them on a validation split, evaluates them with a **leakage-free temporal hold-out and bootstrap confidence intervals**, and serves them through a **Streamlit app** (including live recommendations for brand-new users), a CLI and Docker.

## Highlights
- **Reusable package** (`movierec/`): one interface for every model; saved models load instantly
- **Rigorous evaluation:** latest 20% of each user's ratings held out; hyperparameters tuned on a validation split carved from the *training* data only; 95% **user-level bootstrap CIs** and **paired significance tests** between models
- **Rating and ranking metrics:** RMSE, MAE, Precision/Recall/NDCG/HitRate@10 and catalogue coverage
- **Ablation:** user tags tested as an extra content feature (result: no gain, with the reason documented)
- **Live new-user recommendations** with no retraining; cold-start users fall back to popularity
- **EDA notebook** with executed outputs, **13 automated tests** (including a Streamlit `AppTest`), lint, **GitHub Actions** (3 Python versions, end-to-end run, Docker build) and a **Dockerfile**

## Models
| Model | Type | Idea | Tuned setting |
|---|---|---|---|
| BayesianAverage | Baseline | Item mean rating shrunk toward the global mean | n/a |
| MostPopular | Baseline | Rank by number of ratings | n/a |
| ContentBased | Content | TF-IDF on genres (+ optional user tags); user profile from movies rated above their own average | tag weight 0 |
| ItemKNN | Collaborative | Adjusted-cosine item similarity, co-rating shrinkage, k nearest neighbours, damped weighted average | k=40, shrink=100 |
| SVD | Matrix factorization | Truncated SVD of the user-mean-centred rating matrix | 25 factors |
| Hybrid | Ensemble | Weighted blend of min-max normalised scores | 0.2 SVD, 0.6 ItemKNN, 0.2 Content |

## Quick start
```bash
git clone https://github.com/YOUR-USERNAME/movie-recommender.git
cd movie-recommender
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
make install        # pip install -r requirements-dev.txt
make data           # download MovieLens ml-latest-small (~1 MB)
make all            # lint, tests, tune, evaluate, train, plots, CLI demos, smoke test -> results/ (about 1-2 min)
make app            # Streamlit app at http://localhost:8501
```
Individual steps: `make tune`, `make evaluate`, `make train`, `make report`, `make notebook`, `make test`, `make smoke`.

> Without the dataset the code falls back to a small **synthetic** dataset (with a warning) so tests and demos still run. Report results only from real MovieLens data.

```bash
python -m movierec.cli recommend --user 1 --model Hybrid --k 10
python -m movierec.cli similar --title "Toy Story" --k 10
```

## Streamlit app
| Tab | What it does |
|---|---|
| 👤 Existing user | A user's favourite movies next to the selected model's recommendations |
| 🆕 New user (live) | Pick movies you have seen, rate them, get instant recommendations (no retraining) |
| 🎞 Similar movies | Similar titles by viewing patterns (ItemKNN) and by genres/tags (content) |
| 📊 Model comparison | Metrics table, charts and tuned hyperparameters |

_Add screenshots here (e.g. `docs/app.png`)._

## Docker
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender      # http://localhost:8501
```
The image downloads MovieLens and pre-trains the models at build time (falling back to synthetic demo data if the download fails) and has a health check.

## CI (GitHub Actions)
`.github/workflows/ci.yml` runs on every push and pull request:
1. **test**: lint (`ruff`) and `pytest` on Python 3.10, 3.11 and 3.12
2. **end-to-end**: downloads MovieLens, runs evaluation and training, starts the app and checks it responds
3. **docker**: builds the image, starts the container and checks the health endpoint

## Evaluation methodology
1. Movies with fewer than 5 ratings are removed (raw: 100,836 ratings, 98.3% sparse; after filtering: 3,650 movies, 90,274 ratings, 95.95% sparse).
2. **Temporal split:** each user's latest 20% of ratings form the test set (72,434 train / 17,840 test).
3. **Tuning:** a second temporal split inside the training data gives a validation set. Grid search over SVD factors, ItemKNN (k, shrinkage), tag weight and hybrid weights, selected by validation NDCG@10 (`results/tuning_results.csv`). The test set is never used for tuning.
4. **Tag leakage control:** tags a user applied to movies in the held-out period are removed from features.
5. **Metrics:** RMSE/MAE on all test ratings (rating-scale models only). Ranking metrics @10 treat test ratings >= 4.0 as relevant and rank all movies the user has not seen in training. Coverage is the share of the catalogue recommended to at least one user.
6. **Uncertainty:** 95% bootstrap confidence intervals from 1,000 resamples of users, plus paired bootstrap differences between models.

## Results
Produced by `make all` on real MovieLens data (`results/metrics.csv`, `results/metrics_ci.csv`). Brackets are 95% CIs.

| model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|---|---|---|
| BayesianAverage | 0.973 [0.926, 1.022] | 0.753 | 0.042 | 0.033 | 0.055 [0.046, 0.065] | 0.263 | 0.013 |
| MostPopular | n/a | n/a | 0.057 | 0.051 | 0.076 [0.065, 0.087] | 0.317 | 0.026 |
| ContentBased | n/a | n/a | 0.010 | 0.006 | 0.010 [0.008, 0.014] | 0.090 | 0.604 |
| ItemKNN | **0.921** [0.888, 0.952] | **0.703** | 0.085 | 0.075 | 0.111 [0.096, 0.126] | 0.417 | 0.153 |
| SVD | 0.937 [0.904, 0.967] | 0.721 | 0.072 | 0.063 | 0.097 [0.085, 0.111] | 0.398 | 0.150 |
| **Hybrid** | n/a | n/a | **0.089** | **0.081** | **0.116** [0.101, 0.131] | **0.436** | 0.172 |

![Model comparison](results/model_comparison.png)

**Paired NDCG@10 differences** (`results/paired_comparison.csv`):

| comparison | difference | 95% CI | significant? |
|---|---|---|---|
| Hybrid vs MostPopular | +0.040 | [0.026, 0.054] | yes |
| ItemKNN vs SVD | +0.014 | [0.002, 0.025] | yes |
| Hybrid vs ItemKNN | +0.005 | [-0.002, 0.011] | **no** (CI includes 0) |

### Analysis
- **Personalisation beats popularity.** The Hybrid's NDCG@10 is 53% higher than MostPopular's and statistically significant, and it recommends about 6.5x more of the catalogue (17.2% vs 2.6%).
- **The Hybrid's edge over ItemKNN is not statistically significant**, so the honest conclusion is that ItemKNN and the Hybrid perform similarly; ItemKNN is simpler and also gives rating predictions.
- **Tuning improved ranking but cost rating accuracy.** Selecting by NDCG moved ItemKNN's NDCG from 0.095 (defaults) to 0.111 but its RMSE from 0.899 to 0.921, because heavier similarity shrinkage favours well-known movies. Tuning for RMSE would choose a different setting.
- **Tags did not help.** On validation NDCG the content model scored 0.0095 with no tags, 0.0088 with tag weight 0.5 and 0.0086 with weight 1.0. The EDA explains why: only 16% of movies have any tag, the tags come from just 58 users, and the most common tag is "in netflix queue".
- **Genre-only content filtering is weak** (NDCG 0.010) but has the highest coverage (60%): it ignores quality and popularity. For example, user 1 rated 28 animated films at 4.68 on average (overall mean 4.35), so the model learns "likes animation" and suggests mostly Disney titles.
- **Absolute precision looks low (about 9%)** because each user has only a handful of relevant held-out movies among thousands of candidates, which is typical for temporal splits on MovieLens.

### Limitations
- One temporal split; the CIs capture user sampling variability but not split choice.
- ItemKNN's best shrinkage (100) sits at the edge of the searched grid. Gains were flattening (NDCG 0.106 to 0.107) but larger values were not tried.
- Genres and tags are the only content features; plot or embedding features were not explored.
- `ml-latest-small` has 610 users; results will differ on larger datasets, and offline metrics are proxies for real user satisfaction.
- Trained models (`models/*.joblib`) contain MovieLens-derived data and are **not committed** (see `.gitignore`); run `make train` to create them.

## Execution output
`make all` writes everything to `results/execution_log.txt`. Excerpt:
```text
$ ruff check .                      All checks passed!
$ python -m pytest -v               13 passed
$ python -m movierec.cli tune       Best: SVD 25 factors | ItemKNN k=40 shrink=100 | Hybrid weights [0.2, 0.6, 0.2]
$ python -m movierec.cli evaluate   (results table above, with bootstrap CIs)
$ python -m movierec.cli train      Saved models to models/models.joblib
$ bash scripts/smoke_test.sh        health check OK after 2s / index page OK
```

| File in `results/` | Content |
|---|---|
| `metrics.csv`, `metrics_ci.csv`, `paired_comparison.csv` | Point estimates, bootstrap CIs, significance tests |
| `tuning_results.csv`, `best_params.json` | Full grid-search results and the selected settings |
| `model_comparison.png`, `eda_overview.png` | Charts |
| `sample_recommendations.txt` | Each model's top 10 for a user next to what they really liked later |
| `dataset_summary.json`, `execution_log.txt` | Data statistics and the full run log |

## Project structure
```
movie-recommender/
├── movierec/        # data.py, models.py, evaluate.py, tuning.py, pipeline.py, cli.py
├── app.py           # Streamlit app
├── notebooks/       # 01_eda.ipynb (executed)
├── tests/           # 13 tests: models, evaluation, tuning, persistence, app
├── scripts/         # download_data.py, make_report.py, build_notebook.py, run_all.sh, smoke_test.sh
├── results/         # metrics, CIs, tuning, charts, samples, execution log
├── .github/workflows/ci.yml   Dockerfile   Makefile   ruff.toml   requirements*.txt   LICENSE
```

## Future work
- Implicit-feedback ALS / BPR, neural collaborative filtering; larger MovieLens versions
- Richer content features (plot embeddings), diversity and novelty metrics
- Repeat the evaluation over several temporal splits; tune separately for RMSE
- FastAPI service, online A/B testing

## License
MIT. Dataset © GroupLens Research; see the MovieLens terms of use.
