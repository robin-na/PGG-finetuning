# Raw behavior cluster ablation

This ablation clusters players directly from observed raw behavior rather than LLM summary embeddings.

## Behavior feature vector

- contribution level, zero/full rates, volatility, and endgame delta
- punishment/reward giving and receiving rates and units
- round payoff rate
- chat activity, words per round, and phase-specific message rates when game chat exists

## Downstream evaluation

- same direct benchmark-level prediction target table as the text-cluster analysis
- same linear and ridge regressions
- same deployable (`CONFIG -> predicted cluster mixture`) and oracle (`actual validation mixture`) settings

## Best raw-behavior results

- `cluster_only` best at `K=4`: MAE `0.0836`, RMSE `0.1037`, corr `0.3720`
- `cluster_plus_config` best at `K=4`: MAE `0.0967`, RMSE `0.1277`, corr `0.3262`
- `oracle_cluster_only` best at `K=4`: MAE `0.0736`, RMSE `0.0875`, corr `0.6097`
- `oracle_cluster_plus_config` best at `K=4`: MAE `0.0912`, RMSE `0.1180`, corr `0.4969`

## Best raw-behavior results on mean contribution rate

- `cluster_only` best at `K=8`: MAE `0.0720`, RMSE `0.0867`, corr `0.4602`
- `cluster_plus_config` best at `K=8`: MAE `0.0729`, RMSE `0.0929`, corr `0.3169`
- `oracle_cluster_only` best at `K=8`: MAE `0.0503`, RMSE `0.0649`, corr `0.7463`
- `oracle_cluster_plus_config` best at `K=8`: MAE `0.0590`, RMSE `0.0753`, corr `0.6438`

## Text-cluster reference at K=6

### `mean_contribution_rate`

- `cluster_only` at text `K=6`: MAE `0.0640`, RMSE `0.0824`, corr `0.5439`
- `cluster_plus_config` at text `K=6`: MAE `0.0666`, RMSE `0.0896`, corr `0.4408`
- `cluster_oracle_only` at text `K=6`: MAE `0.0355`, RMSE `0.0447`, corr `0.8861`
- `cluster_oracle_plus_config` at text `K=6`: MAE `0.0386`, RMSE `0.0498`, corr `0.8480`

### `normalized_efficiency`

- `cluster_only` at text `K=6`: MAE `0.0776`, RMSE `0.0961`, corr `0.5620`
- `cluster_plus_config` at text `K=6`: MAE `0.0883`, RMSE `0.1156`, corr `0.4476`
- `cluster_oracle_only` at text `K=6`: MAE `0.0667`, RMSE `0.0913`, corr `0.7006`
- `cluster_oracle_plus_config` at text `K=6`: MAE `0.0618`, RMSE `0.0839`, corr `0.7206`


## Files

- `learn_behavior_features.parquet` / `val_behavior_features.parquet`
- `raw_behavior_cluster_diagnostics.csv`
- `raw_behavior_cluster_direct_eval.csv`
