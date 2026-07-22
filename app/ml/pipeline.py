"""Model pipeline definition.

Preprocessing and the estimator live in a single ``Pipeline`` so that they are
pickled together. Inference can then never drift from training: there is one
artifact, and it contains every transformation applied at fit time.
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.features import HOGTransformer

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Return an unfitted HOG -> scale -> PCA -> RBF SVM pipeline.

    ``class_weight="balanced"`` matters here: the dataset holds roughly three
    non-smiling faces for every smiling one, and an unweighted model reaches a
    respectable accuracy purely by never predicting the minority class.
    """
    return Pipeline(
        steps=[
            ("hog", HOGTransformer()),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )