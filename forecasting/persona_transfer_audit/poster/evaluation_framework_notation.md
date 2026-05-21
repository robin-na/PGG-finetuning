# Evaluation Framework Notation

## Poster-Ready Version

Let `g in G` index target social-interaction games, and let

```text
I_g = {1, ..., n_g}
```

be the real human players observed in game `g`. Each player has a revealed behavior trajectory `b_{gi}` and a pre-specified behavioral feature vector

```text
x_{gi} = f(b_{gi}) in R^d.
```

The empirical human reference distribution within game `g` is uniform over the real observed trajectories:

```text
P_g(i) = 1 / n_g,    i in I_g.
```

For persona source `s`, let `a = 1, ..., m_s` index sampled personas. Given persona `a` and the full transcript of game `g`, the LLM returns a top-`K` distribution over players:

```text
r_{gasi} >= 0,        sum_{i in I_g} r_{gasi} = 1,
```

where unlisted players receive probability `0`. In our experiments, `K = 3`.

Aggregating over personas gives the matched distribution for persona source `s` in game `g`:

```text
Q_{gs}(i) = (1 / m_s) sum_{a=1}^{m_s} r_{gasi}.
```

For the no-persona baseline, `m_s = 1`.

For behavioral feature `l`, the human and matched feature means are:

```text
mu^P_{gl}  = sum_{i in I_g} P_g(i) x_{gil}

mu^Q_{gsl} = sum_{i in I_g} Q_{gs}(i) x_{gil}.
```

The game-level behavioral skew is the matched-minus-human difference:

```text
Delta_{gsl} = mu^Q_{gsl} - mu^P_{gl}.
```

The plotted behavioral skewness statistic aggregates across target games and standardizes by the empirical human standard deviation of that feature:

```text
Delta_tilde_{sl}
  = [ (1 / |G|) sum_{g in G} Delta_{gsl} ] / sigma^P_l,
```

where

```text
sigma^P_l = SD_{g in G, i drawn from P_g}(x_{gil}).
```

Interpretation:

```text
Delta_tilde_{sl} = 0
```

means that persona source `s` matches the empirical human mean for feature `l`. Positive values mean the LLM-persona system selects players with higher values of that behavior than the human reference distribution; negative values mean it selects players with lower values.

## Figure Caption Draft

Behavioral skewness of persona-conditioned matches. For each target game, the empirical human reference distribution `P_g` is uniform over all real players observed in that game. For each persona source `s`, the matched distribution `Q_{gs}` is obtained by averaging the LLM's top-3 probability distributions over sampled personas. Each cell shows the matched-minus-human difference in a behavioral feature, standardized by the empirical human standard deviation of that feature. Values near zero indicate that the selected trajectories match the human reference distribution on that feature; positive and negative values indicate over-selection and under-selection, respectively.

## Equation-Editor Lines

Paste each line into a Microsoft Word or PowerPoint equation box.

```text
I_g = \{1,\ldots,n_g\}
```

```text
x_{gi}=f(b_{gi})\in\mathbb{R}^d
```

```text
P_g(i)=\frac{1}{n_g},\quad i\in I_g
```

```text
r_{gasi}\ge 0,\quad \sum_{i\in I_g} r_{gasi}=1
```

```text
Q_{gs}(i)=\frac{1}{m_s}\sum_{a=1}^{m_s} r_{gasi}
```

```text
\mu^P_{g\ell}=\sum_{i\in I_g}P_g(i)x_{gi\ell}
```

```text
\mu^Q_{gs\ell}=\sum_{i\in I_g}Q_{gs}(i)x_{gi\ell}
```

```text
\Delta_{gs\ell}=\mu^Q_{gs\ell}-\mu^P_{g\ell}
```

```text
\widetilde{\Delta}_{s\ell}=
\frac{\frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}\Delta_{gs\ell}}
{\sigma^P_\ell}
```

```text
\sigma^P_\ell=\operatorname{SD}_{g\in\mathcal{G},\, i\sim P_g}(x_{gi\ell})
```
