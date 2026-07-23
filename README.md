# Smile Classifier - 30225

Classifies a face photo as smiling or not smiling. FastAPI, scikit-learn,
PostgreSQL, Docker.

## Quick start

Requires Docker Desktop.

```bash
docker network create YSDTP_B5_AI_30225
cp .env.example .env
docker compose up -d --build
```

Open <http://localhost:8000>. A trained model ships inside the image, so
classification works immediately — no training required first.

| Service | URL |
|---|---|
| Application | http://localhost:8000 |
| pgAdmin | http://localhost:5051 |

pgAdmin credentials come from `.env`. When adding the server, use host `db`
and port `5432` — inside that container `localhost` means pgAdmin itself.

Sample photos for testing are inside the container at `/app/data/samples`,
and in this repository under `seed/samples/`.

### Port conflicts

If a port is already in use, change it in `.env` and re-run
`docker compose up -d`. Only the host side changes; nothing inside the
containers is affected.

| Variable | Default | Used by |
|---|---|---|
| `APP_PORT` | 8000 | Application |
| `POSTGRES_PORT_HOST` | 15432 | PostgreSQL |
| `PGADMIN_PORT` | 5051 | pgAdmin |

## Pages

- **Home** — how the model works, and which model is currently active
- **Train** — upload labelled photos and fit a new model from them
- **Classify** — upload one photo, get a verdict and confidence
- **History** — every classification, newest first

## How it works

Images are converted to greyscale and resized to 64x64. Colour carries no
information about whether someone is smiling, and dropping it removes two
thirds of the input for free.

Each image becomes a **Histogram of Oriented Gradients** descriptor: the image
is split into 8x8 cells and each cell records the directions in which
brightness changes. A smile is a specific arrangement of curved edges around
the mouth, which is what this measures, and it is far more tolerant of
lighting than raw pixels because it looks at how brightness *changes* rather
than what it is.

HOG is implemented directly in numpy rather than imported from scikit-image.
It is one well-defined algorithm, and writing it out removed a large compiled
dependency with no measurable loss in accuracy.

The 1,764 resulting values pass through standardisation, PCA retaining 95% of
variance, then an **RBF-kernel support vector machine**. With around 1,200
training images and more feature dimensions than samples, an SVM is the right
classical choice: it optimises the margin between the two classes rather than
trying to model the whole space. A neural network would need far more data, a
GPU, and a much larger container to solve a problem that fits in two seconds
of CPU time.

Preprocessing and classifier are wrapped in a single scikit-learn `Pipeline`
and pickled together, so inference cannot drift from training: there is one
artifact, and it contains every transformation applied at fit time.

### Results

Trained on the Kaggle `chazzer/smiling-or-not-face-data` set: 600 smiling,
603 not smiling.

| Metric | Value |
|---|---|
| Accuracy | 94.61% |
| F1 (smiling) | 0.9465 |
| Validation | 20% stratified holdout |
| Training time | ~17s on CPU |

The dataset is close to balanced, so accuracy is meaningful on its own. F1 for
the smiling class is reported beside it as a check: if the two diverge, the
model has begun favouring one class and accuracy has stopped meaning what it
appears to.

## Retraining through the app

The Train page takes photos one class at a time, because a batch carries a
single label and a two-class model cannot be fit from one class. Photos
accumulate in a staging area until both classes hold at least five, at which
point training can start.

Training replaces the active model with one fit to the uploaded photos, as
specified. Accuracy will be much lower than the shipped model, because a
handful of images cannot teach what 1,200 did — that is inherent to the
approach, not a fault. A production system would accumulate uploads into the
existing corpus and retrain on everything.

Training runs in the background, so the browser is never left waiting. The
previous model keeps serving until a new one has been trained, written to
disk, read back and validated. A failed run is therefore a non-event: it is
recorded on the Train page and the working model stays in service.

The app holds the model in memory and compares the file's modification time
before each prediction, so a retrain takes effect on the very next
classification with no restart.

## Architecture

The ML core in `app/ml/` has no web or database imports. `scripts/train.py`
and the Train page both call the same `train_from_directory`, so there is one
definition of what training means.

`HOGTransformer` lives in `app/ml/features.py` rather than in a script.
Pickle stores only a class's import path, so a transformer defined in
`__main__` cannot be unpickled by the web process.

## Command line

```bash
python scripts/bootstrap_dataset.py <extracted kaggle folder>
python scripts/train.py
python scripts/predict.py photo.jpg
```

## Local development

Requires Python 3.14.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
Copy-Item seed\smile_clf.pkl models\smile_clf.pkl
docker compose up -d db
uvicorn app.main:app --reload --port 8000
```

Set `POSTGRES_HOST=localhost` and `POSTGRES_PORT` to match
`POSTGRES_PORT_HOST` in `.env` when running outside Docker.

## Notes

- `docker compose down` is safe. `docker compose down -v` **deletes all data**,
  including classification history.
- Uploads are re-encoded to JPEG through Pillow, which doubles as validation:
  a file that is not really an image fails to decode and never reaches disk.
- The container runs a single uvicorn worker by design. The model cache and
  background training tasks live in process memory, so multiple workers would
  serve stale models after a retrain. Scaling out would move training to a
  dedicated worker consuming a job queue.
- The database schema is created with SQLAlchemy `create_all`. A longer-lived
  system would use Alembic migrations.