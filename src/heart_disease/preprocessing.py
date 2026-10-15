"""
Scikit-learn preprocessing pipeline builder.

Responsibilities
----------------
* Build a ``ColumnTransformer`` that handles numeric and categorical features
  separately with proper imputation and scaling/encoding.
* Return a full sklearn ``Pipeline`` with the preprocessor as the first step
  and a placeholder for a classifier as the second step.
* All transformers are fitted **only on training data** — never on the test set.

Design decisions
----------------
Numeric features  → ``SimpleImputer(strategy="median")`` → ``StandardScaler()``
Categorical features → ``SimpleImputer(strategy="most_frequent")``
                     → ``OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)``

``OrdinalEncoder`` is used instead of ``OneHotEncoder`` for tree-based models
(Random Forest) because RF does not require one-hot encoding and ordinal
representation avoids the explosion of feature dimensionality.
For SVM the same pipeline is reused; the StandardScaler already applied to
numeric features means categorical ordinal codes are bounded small integers,
which the RBF kernel handles well in practice. If one-hot is needed for a
specific model, pass ``use_onehot=True`` to ``build_pipeline``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from heart_disease.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(use_onehot: bool = False) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` for the Heart Disease feature set.

    Parameters
    ----------
    use_onehot : bool
        If ``True``, encode categorical features with ``OneHotEncoder``
        (suitable for linear / kernel models that benefit from binary inputs).
        If ``False`` (default), use ``OrdinalEncoder`` (suitable for tree-based
        models and saves memory).

    Returns
    -------
    ColumnTransformer
        Unfitted transformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    if use_onehot:
        # Build explicit category lists in CATEGORICAL_FEATURES order so that
        # OneHotEncoder knows every valid value upfront. This prevents
        # "unknown categories" warnings during cross-validation folds where a
        # rare category (e.g. restecg=2) might not appear in the training fold.
        from heart_disease.config import FEATURE_VALID_VALUES
        ohe_categories = [
            sorted(FEATURE_VALID_VALUES[f]) for f in CATEGORICAL_FEATURES
        ]
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                categories=ohe_categories,
                handle_unknown="ignore",
                drop="if_binary",
                sparse_output=False,
            )),
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",       # drop any extra columns silently
        verbose_feature_names_out=True,
    )
    return preprocessor


def build_pipeline(classifier, use_onehot: bool = False) -> Pipeline:
    """Wrap the preprocessor and a classifier into a single ``Pipeline``.

    The classifier is inserted under the key ``"classifier"`` so that
    ``GridSearchCV`` can target its parameters with the ``classifier__``
    prefix (e.g. ``classifier__n_estimators``).

    Parameters
    ----------
    classifier : sklearn estimator
        An unfitted scikit-learn classifier with a ``fit`` / ``predict``
        interface. Must support ``predict_proba`` for probability fusion.
    use_onehot : bool
        Passed through to ``build_preprocessor``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline: preprocessor → classifier.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(use_onehot=use_onehot)),
        ("classifier",   classifier),
    ])
