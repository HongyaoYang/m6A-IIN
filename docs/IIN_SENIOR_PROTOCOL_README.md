# IIN training and OOD protocol aligned to the senior iM6A scripts

These scripts use the audited author-code-faithful architecture in
`m6a_iin_forward_regression.py`. The architecture is fixed to author repository
commit `16a02193855b325538872b52df1331f88e32cea1`; the regression protocol is
kept separately in the training/evaluation scripts.

## Data layout

The default paths match the senior scripts:

```text
/gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT/
├── sequences.fa
├── train_site.csv
├── valid_site.csv
├── DART_seq.csv
├── eTAM_seq.csv
└── ... one ground-truth CSV per technology
```

`train_site.csv` and `valid_site.csv` must contain `tech` and `site_idx`, where
each `site_idx` cell is a comma-separated list. Each technology ground-truth
CSV must contain `site_idx,ratio,m6A,Total,class`.

The supplied archive is RAR v5 (despite earlier references to tar). On the HPC,
extract and validate it with:

```bash
bash prepare_tech_GT_on_HPC.sh \
  /gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT.rar \
  /gpfs/work/bio/hongyaoyang23/m6A_IIN
```

This writes `tech_GT/IIN_data_contract_report.json`. The checker scans the
large CSVs in chunks and verifies columns, count constraints, ratio consistency,
split overlap/membership, coverage-dependent class labels, and FASTA IDs.
When R is available it also writes `tech_GT/IIN_RDS_summary.txt`, including the
train/valid/test sizes and any sequence-like columns present in `grg.rds`.

Important: the supplied RAR does not contain `sequences.fa`. The IIN pipeline
cannot construct its 41-nt inputs until this FASTA is supplied or generated.
FASTA headers must begin with integer `site_idx` values.

## 0. Precompute the author's ELMo input

The author does not train a new ELMo model inside IIN and does not distribute an
ELMo feature cache. Their preprocessing calls Flair `ELMoEmbeddings('small')`.
Historical Flair maps this name to AllenNLP's
`2x1024_128_2048cnn_1xhighway` model, concatenates its three 256-dimensional
representations to 768 dimensions, and the author then sums the three slices to
256 dimensions.

Official files:

- `elmo_2x1024_128_2048cnn_1xhighway_options.json`
- `elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5`

First verify the author-style embedding on one 41-nt sequence:

```bash
python verify_author_elmo.py \
  --options-file tech_GT/elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json \
  --weight-file tech_GT/elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 \
  --device cuda
```

Only after this prints `PASS`, generate the shared feature array once:

```bash
python precompute_author_elmo.py \
  --fasta tech_GT/sequences_41nt.fa \
  --output tech_GT/author_elmo_features.npy \
  --options-file tech_GT/elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json \
  --weight-file tech_GT/elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 \
  --batch-size 64 \
  --device cuda
```

For 2,526,215 sites, the float32 feature array is approximately 106 GB. The
script is resumable through paired `.partial.npy` files. Batched embedding is an
execution optimization; the per-sequence operation and 768-to-256 reduction are
the same as the author's preprocessing.

Run the small architecture check before formal training:

```bash
python verify_author_faithful_IIN.py
```

## 1. Train one technology

```bash
python train_IIN.py --tech DART_seq
```

Run once for each of the 11 technologies. Training and validation use every
listed site with `Total > 0`. No site is dropped from a short final batch.
The only training objective is stable Binomial NLL. The default protocol is:

- scalar logit followed by sigmoid mu;
- Adam, learning rate 1e-3;
- batch size 128;
- 20 epochs;
- fixed random seed 2023, matching the released source;
- best checkpoint selected by validation Binomial NLL over all `Total > 0`
  validation sites.

Validation reports AUROC, AUPRC, R2, Spearman rho and Pearson cor. The reporting
coverage threshold defaults to 10 and does not affect checkpoint selection.

## 2. Predict all eligible sites

After all checkpoints have been trained:

```bash
python OOD_IIN.py --tech all
```

For each source model, this predicts every FASTA site except that source
technology's own train and validation IDs. No Total threshold is applied during
inference. Every prediction is saved as:

```text
<technology>_allSite_preds.csv
```

with `model_technology,site_idx,pred` columns.

OOD graph construction is chunked (10,000 sites by default) to avoid loading
all 41-by-41 graph matrices into memory at once. This changes only the inference
memory usage, not the IIN model or prediction calculation.

## 3. Calculate OOD metrics later

```bash
python evaluate_OOD_IIN.py --min-total 10
```

This joins every source model's saved predictions to each target technology's
ground truth. Only at this stage are rows filtered to `Total >= 10`. Outputs:

- `IIN_OOD_metrics.csv`: source-by-target results and sample counts;
- `IIN_OOD_metrics_summary.csv`: mean and standard deviation across true
  cross-technology OOD pairs only.

Rows with the same source and target are labelled `same_technology_holdout`,
not OOD. Cross-technology rows are labelled `cross_technology_OOD`.

## Important boundary

The IIN input representation supports A, C, G and U/T. A requested 41-nt window
containing N or another ambiguous base raises an error rather than silently
changing the author's 15-dimensional nucleotide representation.

Old checkpoints trained with the earlier internal BiLSTM/loop/GWNN/gate
implementation are architecture-incompatible and must not be mixed with this
version. Retrain all 11 technologies after ELMo preprocessing and the smoke test
have passed.
