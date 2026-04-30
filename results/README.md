# Results Directory Notes

Current source of truth:
- `model_comparison.csv`
- `*_evaluation_results.csv`

Historical files:
- Older markdown analysis reports in this directory were generated from earlier experiment runs.
- Some of them contain outdated metrics or encoding issues and should not be used as the final paper source.

Recommended workflow:
1. Run `train_all_models.py` for the full experiment suite.
   - Example: `python train_all_models.py`
   - Single/partial run: `python train_all_models.py --models twotower deepfm`
   - List available models: `python train_all_models.py --list-models`
2. Use `results/model_comparison.csv` as the master comparison table.
3. Use the latest `results/*_evaluation_results.csv` files for per-model metrics.
