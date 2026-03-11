# Wave-Anchored OOD Direct Benchmark Results

This report evaluates direct macro predictors under true wave-anchored one-factor OOD splits built from `benchmark_statistical/data`.

Setup:
- text-cluster model uses `K=6` and refits the GMM + Dirichlet env model on each split train set
- raw-behavior model uses `K=4` and refits the GMM + Dirichlet env model on each split train set
- all reported main results below are deployable ridge models, not oracle mixtures

## Mean across OOD splits

### `contribution_rate`

- `ridge_config`: mean MAE `0.0780`, mean RMSE `0.0982`, mean corr `0.3886` across `18` splits
- `ridge_raw_cluster_pred_cluster_plus_config`: mean MAE `0.0813`, mean RMSE `0.1008`, mean corr `0.3316` across `18` splits
- `ridge_raw_cluster_pred_cluster_only`: mean MAE `0.0815`, mean RMSE `0.0984`, mean corr `0.2563` across `18` splits
- `ridge_text_cluster_pred_cluster_plus_config`: mean MAE `0.0815`, mean RMSE `0.0996`, mean corr `0.3785` across `18` splits
- `ridge_text_cluster_pred_cluster_only`: mean MAE `0.0858`, mean RMSE `0.1052`, mean corr `0.3358` across `18` splits

### `normalized_efficiency`

- `ridge_raw_cluster_pred_cluster_only`: mean MAE `0.0995`, mean RMSE `0.1226`, mean corr `0.2347` across `18` splits
- `ridge_text_cluster_pred_cluster_only`: mean MAE `0.1068`, mean RMSE `0.1303`, mean corr `0.3384` across `18` splits
- `ridge_text_cluster_pred_cluster_plus_config`: mean MAE `0.1119`, mean RMSE `0.1356`, mean corr `0.3747` across `18` splits
- `ridge_config`: mean MAE `0.1142`, mean RMSE `0.1378`, mean corr `0.4019` across `18` splits
- `ridge_raw_cluster_pred_cluster_plus_config`: mean MAE `0.1172`, mean RMSE `0.1431`, mean corr `0.3221` across `18` splits
