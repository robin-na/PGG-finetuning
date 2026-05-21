# Compact Evaluation Notation

Use this version on the poster.

For each game `g`, let real players be `i = 1, ..., n_g`, with observed behavioral features `x_{gi}`.

**Human baseline**

```text
P_g(i) = 1 / n_g
```

**LLM-persona matched distribution**

Persona `a` from source `s` assigns top-3 probabilities `r_{gasi}` over observed players. Average across sampled personas:

```text
Q_{gs}(i) = Avg_a r_{gasi}
```

**Behavioral skewness plotted**

```text
delta_tilde_{sl} =
(1 / sigma^P_l) Avg_g { E_{Q_{gs}}[x_l] - E_{P_g}[x_l] }
```

Interpretation: `0` means matched trajectories align with the human baseline for behavior `l`; positive/negative values mean over-/under-selection of that behavior.
