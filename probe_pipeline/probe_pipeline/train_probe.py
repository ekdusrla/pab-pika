#!/usr/bin/env python3
"""Train the credit-prediction probe from probe_training_data.pt.

Unlike the earlier version of this script, there is NO teacher-forcing pass
here anymore: generate_training_data.py already saved the hidden state for
every layer, captured live during STEP 1's generation. This script only
needs to load that file and fit a regressor per layer.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path('probe_training_data.pt')
OUT_DIR = Path('probe_artifacts')


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = torch.load(DATA_PATH, weights_only=False)
    print(f'Loaded {len(rows)} training pairs from {DATA_PATH}', flush=True)

    n_layers = len(rows[0]['hidden_states'])
    y = np.array([r['credit'] for r in rows], dtype=float)

    best_layer, best_score, best_pipeline = None, -np.inf, None
    for layer_idx in range(n_layers):
        X = np.stack([r['hidden_states'][layer_idx] for r in rows])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        reg = Ridge(alpha=1.0)
        scores = cross_val_score(
            reg, X_scaled, y, cv=KFold(5, shuffle=True, random_state=0), scoring='r2'
        )
        mean_r2 = scores.mean()
        print(f'  layer {layer_idx:>2}: R^2 = {mean_r2:.4f}', flush=True)
        if mean_r2 > best_score:
            best_score = mean_r2
            best_layer = layer_idx
            reg.fit(X_scaled, y)
            best_pipeline = {'scaler': scaler, 'regressor': reg}

    print(f'\nBest layer: {best_layer} (R^2 = {best_score:.4f})', flush=True)

    joblib.dump(best_pipeline, OUT_DIR / 'best_probe.joblib')
    (OUT_DIR / 'probe_metadata.json').write_text(json.dumps({
        'base_model': 'LLaMA-3.1-8B-Instruct (local)',
        'best_layer_idx': best_layer,
        'cv_r2': best_score,
        'n_training_pairs': len(rows),
        'step_marker': '<<<STEP>>>',
        'note': 'At inference, call generate_with_step_hidden_states() and '
                'use ONLY hidden_states[best_layer_idx] with this probe.',
    }, indent=2), encoding='utf-8')

    print(f'Saved: {OUT_DIR}/best_probe.joblib, {OUT_DIR}/probe_metadata.json', flush=True)


if __name__ == '__main__':
    main()
