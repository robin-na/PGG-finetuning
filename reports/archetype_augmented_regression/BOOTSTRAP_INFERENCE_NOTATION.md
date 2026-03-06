# Bootstrap Inference Notation (CONFIG-treatment level)

Let \(x \in \{1,\dots,N\}\) index CONFIG-treatment groups (single run: \(N=40\)).

- Observed validation group mean target: \(y_x\)
- Model-\(m\) prediction: \(\hat y_{m,x}\)
- Error: \(e_{m,x}=y_x-\hat y_{m,x}\)
- Learning-wave mean target at this granularity: \(\mu_{\text{train}}\)

Metrics for model \(m\):

\[
\mathrm{RMSE}_m=\sqrt{\frac{1}{N}\sum_{x=1}^N e_{m,x}^2}
\]

\[
R^2_{m,\text{test}}=1-\frac{\sum_{x=1}^N e_{m,x}^2}{\sum_{x=1}^N (y_x-\bar y)^2},
\quad
\bar y=\frac{1}{N}\sum_{x=1}^N y_x
\]

\[
R^2_{m,\text{train}}=1-\frac{\sum_{x=1}^N e_{m,x}^2}{\sum_{x=1}^N (y_x-\mu_{\text{train}})^2}
\]

Sampling-noise ceiling (from the PDF definition):

- Within group \(x\): sample variance \(s_x^2\), count \(n_x\)
- Noise term \(v_x=s_x^2/n_x\)

\[
\mathrm{MSE}_{\text{floor}}=\frac{1}{N}\sum_{x=1}^N v_x,
\quad
\mathrm{RMSE}_{\text{floor}}=\sqrt{\mathrm{MSE}_{\text{floor}}}
\]

\[
R^2_{\text{ceil,test}}=1-\frac{\mathrm{MSE}_{\text{floor}}}{\frac{1}{N}\sum_{x=1}^N (y_x-\bar y)^2},
\quad
R^2_{\text{ceil,train}}=1-\frac{\mathrm{MSE}_{\text{floor}}}{\frac{1}{N}\sum_{x=1}^N (y_x-\mu_{\text{train}})^2}
\]

Paired bootstrap:

- Draw \(N\) groups with replacement, \(b=1,\dots,B\).
- Recompute all model metrics on the same resample (paired design).
- 95% CI is percentile \([q_{0.025}, q_{0.975}]\) across bootstrap draws.

Paired deltas (A vs B):

\[
\Delta R^2_{\text{test}}=R^2_{A,\text{test}}-R^2_{B,\text{test}}
\]

\[
\Delta R^2_{\text{train}}=R^2_{A,\text{train}}-R^2_{B,\text{train}}
\]

\[
\Delta \mathrm{RMSE}_{\text{drop}}=\mathrm{RMSE}_B-\mathrm{RMSE}_A
\]

Positive \(\Delta\) means model A is better.
