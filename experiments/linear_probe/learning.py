"""StandardScaler + Ridge; problem-level train/validation and frozen test."""
import time

from .common import FEATURE, credit_contract, digest, project_credits


def feature_rows(row, layers=None):
    features = row['capture']['features']
    if isinstance(features, list):
        return features  # Synthetic CPU tests only; CLI stores compressed arrays.
    from .common import file_hash
    import numpy as np
    if file_hash(features['path']) != features['sha256']:
        raise ValueError('Feature file changed')
    with np.load(features['path'], allow_pickle=False) as arrays:
        selected = arrays.files if layers is None else [str(layer) for layer in layers]
        loaded = {layer: arrays[layer] for layer in selected}
        if any(a.ndim != 2 or len(a) != features['n_steps'] for a in loaded.values()):
            raise ValueError('Feature archive shape mismatch')
        return [{layer: a[i] for layer, a in loaded.items()} for i in range(features['n_steps'])]


def validate_capture(doc, manifest, split):
    from .common import verify
    verify(doc)
    if doc['data_fingerprint'] != manifest['fingerprint'] or doc['split'] != split:
        raise ValueError('Capture/data/split mismatch')
    expected = {r['id']: r for r in manifest['splits'][split]}
    rows = doc['records']
    if len(rows) != len(expected) or {r['id'] for r in rows} != set(expected):
        raise ValueError('Require complete unique split; no silent subset training/evaluation')
    for row in rows:
        if row['problem_hash'] != expected[row['id']]['problem_hash']:
            raise ValueError('Problem identity mismatch')
        if row['status'] != 'success' or row['capture']['feature_definition'] != FEATURE:
            raise ValueError('Incomplete capture or incompatible features')
        cap = row['capture']
        if len(cap['steps']) != len(feature_rows(row)) or not 2 <= len(cap['steps']) <= 5:
            raise ValueError('Feature/step mismatch')
        if row.get('steps_fingerprint') != digest(cap['steps']):
            raise ValueError('Changed decomposition')
    return rows


def matrix(rows, layer):
    import numpy as np
    X, y = [], []
    for row in rows:
        credits = row['teacher']['credits']
        credit_contract(credits, len(row['capture']['steps']))
        for feature, credit in zip(feature_rows(row, [layer]), credits):
            X.append(feature[str(layer)]); y.append(credit)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite or invalid training features')
    return X, y


def predict(pipeline, row, layer):
    import numpy as np
    begin = time.perf_counter()
    X = np.asarray([f[str(layer)] for f in feature_rows(row, [layer])], dtype=float)
    read_seconds = time.perf_counter() - begin
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('Invalid feature matrix')
    begin = time.perf_counter()
    raw = pipeline.predict(X)
    inference_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    projection = project_credits(raw)
    projection['inference_seconds'] = inference_seconds
    projection['feature_read_seconds'] = read_seconds
    projection['projection_seconds'] = time.perf_counter() - begin
    projection['logical_generation_calls'] = 0
    projection['prompt_tokens'] = projection['completion_tokens'] = 0
    projection['extra_forward_passes'] = 0
    return projection


def fit_probe(train, validation, layers, alpha=1.0):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if not layers or alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('Need layers and positive finite alpha')
    scores, best = [], None
    for layer in sorted(layers):
        X, y = matrix(train, layer)
        pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha))])
        pipeline.fit(X, y)  # Neither scaler nor regressor sees validation/test.
        errors = []
        for row in validation:
            pred = predict(pipeline, row, layer)['credits']
            errors.append(float(np.mean(np.abs(np.asarray(pred) - row['teacher']['credits']))))
        score = float(np.mean(errors))  # Equal weight per problem, not per sub-question.
        scores.append({'layer': layer, 'validation_problem_mean_credit_mae': score})
        if best is None or score < best[0]:
            best = (score, layer, pipeline)
    # Do not refit on validation: keep its role as an explicit selection set.
    return best[2], {'best_layer_idx': best[1], 'alpha': alpha, 'layer_selection': scores,
                     'selection_metric': 'validation problem-mean projected-credit MAE',
                     'validation_used_for_fit': False}
