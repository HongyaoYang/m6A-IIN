# m6A-IIN author-fidelity audit

## Declared baseline

- Author repository: `ZLSRW/m6A-IIN`
- Audited commit: `16a02193855b325538872b52df1331f88e32cea1`
- Architecture source files: `0.Structure-Graph.py`, `1.feature_embedding.py`,
  `GraphInteraction.py`, `Graph_Wavelet.py`, `ELMom6A.py`, and `IBGNN.py`
- Paper: *m6A-IIN: identifying m6A modification sites and associated biological
  mechanisms by an invertible neural network with a reverse biological
  interpretability*, Communications Biology (2025) 8:1022.

## Audit result

| Component | Current decision | Status |
|---|---|---|
| 15-D sequence encoding | Copy released lookup values | Author code |
| Secondary graph | Copy bracket pairing and diagonal connections | Author code |
| Loop graph | Copy released `filter_elements` heuristic literally | Author code |
| LRP | Copy Gaussian `sigma=0.5`, MLP and BatchNorm order | Author code |
| NCM | Copy attention, BatchNorm, LeakyReLU and normalization order | Author code |
| ELMo input | Use pretrained Flair `ELMoEmbeddings('small')`; sum 3 x 256 slices | Author code |
| Semantic network | Copy Conv/BN/dropout and two unidirectional LSTMs averaged | Author code |
| GWNN | Copy two layers and fast Chebyshev wavelet (`0.3`, `1e-4`, order `3`) | Author code |
| Regional/inverse branch | Copy scaling matrix, first 15 columns, `15->15`, inverse coupling | Author code |
| Custom primary gate | Removed | Previous local extension |
| Output | Keep author `271->1`; remove terminal sigmoid and expose logit | Regression adaptation |
| Objective | Binomial NLL only on `Total>0` | Senior protocol |
| Split/checkpoint | Fixed train/valid; minimum validation Binomial NLL | Senior protocol |
| OOD | Save all predictions; metrics only at `Total>=10` | Senior protocol |

## Minimal executable corrections

1. The released coupling layer returns `torch.tensor(B1)` and
   `torch.tensor(B2)`, which copies/detaches existing tensors. The audited version
   directly returns `B1, B2`, preserving the same values and autograd graph.
2. Released helper code hard-codes `cuda:0`. The audited version uses the device
   of the input/model so it can run on the allocated GPU or CPU.
3. ELMo records are embedded in batches and stored as float32 memory-mapped
   arrays. The token-level model and the author's three-slice summation are not
   changed.

No biological reinterpretation, alternative loop definition, replacement
heat-kernel GWNN, or new dimensional projection is included.

## Remaining non-equivalences that must be reported

1. This is a quantitative stoichiometry regression task, while the paper's main
   experiment is balanced binary classification.
2. The paper describes two BLSTM layers, whereas the released `ELMom6A.py`
   contains two separate unidirectional LSTMs applied to the same tensor and
   averages them. Following the requested source priority, current code follows
   the released code.
3. The paper trains with BCE plus regional-attention and inverse losses. The
   agreed current protocol uses Binomial NLL only. The inverse path executes,
   but inverse-only parameters are not optimized by an unused auxiliary loss.
4. The author specifies Python 3.7, PyTorch 1.8.0 and NumPy 1.19.5. The target
   HPC currently uses newer software, so exact bitwise reproduction is not
   promised.
5. The author does not pin Flair/AllenNLP versions or distribute the ELMo cache.
   Historical Flair confirms that `ELMoEmbeddings('small')` selects AllenNLP's
   `2x1024_128_2048cnn_1xhighway` options and weights. A separate author-era
   environment and the supplied one-sequence check are still required before
   the full 106 GB feature generation.

## Operational decision

All checkpoints produced by the superseded mixed architecture, including the
earlier MeRIP model, must be kept separate and must not be used in the final
11-technology comparison. Formal retraining starts only after the ELMo and
architecture smoke tests pass.
