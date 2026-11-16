# IIN low-R2 tuning workflow

This workflow tunes only `MeRIP_seq`, `scDART_seq`, `m6ACE_seq`, and
`MAZTER_seq`. It does not change the IIN architecture, 41-nt input, ELMo and
RNAfold caches, fixed train/validation split, Binomial NLL, or Total threshold.

## Search space

- Adam, initial learning rate `3e-4`
- Adam, initial learning rate `1e-4`
- RMSProp, initial learning rate `8e-5`
- batch size 128 and seed 2023 for every run
- maximum 60 epochs
- validation-NLL plateau scheduler, factor 0.3, patience 5, minimum LR `1e-6`
- global gradient clipping at 5.0
- early stopping after 15 validation epochs without NLL improvement

The winner for each technology is selected exclusively by minimum validation
Binomial NLL. Validation R2 is retained as a diagnostic but is not the selection
criterion. Test and OOD results must not be used to choose a configuration.

## Outputs

All new results are isolated under:

`model_IIN/tuning_low_r2/`

The workflow writes:

- `tuning_validation_summary.csv`: every completed tuning run;
- `selected_by_validation_nll.csv`: one validation-selected candidate per technology;
- `.complete/`: restart-safe completion markers;
- `TUNING_COMPLETE`: created only after the complete search finishes.

Existing checkpoints under `model_IIN/final_author_elmo/` and their OOD
predictions are never overwritten. After reviewing the selected validation
results, build a separate final candidate directory and recompute only the OOD
prediction rows whose source models changed.
