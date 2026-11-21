"""Finite-sample power using the deployed simultaneous exact paired interval."""
import json

import numpy as np
from scipy.stats import beta, binom


def power(n, discordance, delta, margin=.03):
    if abs(delta) > discordance:
        return None
    counts = np.arange(n+1)
    lower = np.zeros(n+1)
    upper = np.ones(n+1)
    lower[1:] = beta.ppf(.0125, counts[1:], n-counts[1:]+1)
    upper[:-1] = beta.ppf(.9875, counts[:-1]+1, n-counts[:-1])
    win_conditional = (discordance+delta)/(2*discordance)
    prob = 0.
    for discordant, weight in enumerate(binom.pmf(counts,n,discordance)):
        wins = np.arange(discordant+1)
        passes = lower[wins] - upper[discordant-wins] > -margin
        prob += weight * binom.pmf(wins,discordant,win_conditional)[passes].sum()
    return float(prob)


if __name__ == '__main__':
    scenarios = [dict(discordance=q,true_difference=delta,power=power(500,q,delta))
                 for q in (.02,.05,.10,.20,.30) for delta in (-.03,-.01,0,.01)]
    print(json.dumps(dict(n=500, margin_pp=3, target_power=.8,
                          decision='two-sided 95% exact paired conservative lower bound > -3pp',
                          scenarios=scenarios), indent=2, allow_nan=False))
