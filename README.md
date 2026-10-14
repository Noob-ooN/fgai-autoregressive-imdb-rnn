# Autoregressive IMDb Text Generation with Recurrent Neural Networks

Foundations of Generative AI (ICT4403), Assignment 2

## Problem statement and objective

Build a word-level autoregressive language model that predicts the next token from a preceding context and generates movie-review text sequentially. Compare Simple RNN, Bidirectional RNN, LSTM, and GRU under a shared experimental protocol. Evaluate predictive performance, training cost, generated-text quality, and sampling temperatures 0.2, 0.7, and 1.2.

## Dataset and source

The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) by Maas et al. contains 50,000 labelled IMDb reviews split into 25,000 training and 25,000 test reviews, plus additional unlabelled data. Download: [original Stanford archive](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz). Reference: Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL.

This experiment randomly selects 5,000 reviews from the official labelled training partition: 2,500 positive and 2,500 negative. Sentiment labels balance selection but are discarded for language modelling. The selected reviews are shuffled with seed 42 and split by review into 4,000 training, 500 validation, and 500 held-out test reviews. These are custom splits of the official training partition; the official test and unlabelled partitions are unused.

Reviews contain informal English, movie names, HTML breaks, variable lengths, and polarized opinions. Truncating reviews and limiting the vocabulary makes computation manageable but removes long-range context and maps rare names to an unknown token.

## Part 1: Preprocessing and sequence construction

1. Decode HTML entities, remove HTML tags, and lowercase text.
2. Tokenize words, contractions, numbers, and selected punctuation. Punctuation is retained as a token, so this is word-level modelling with punctuation tokens.
3. Retain at most 80 tokens per review. Add beginning/end tokens.
4. Fit a vocabulary of at most 4,000 entries **only on training reviews**, including PAD, UNK, BOS, and EOS.
5. Construct fixed-length 16-token input contexts with one next-token target, using a stride of four target positions. Left-pad short prefixes. Never cross review boundaries.

For illustration with stride one, `the film was good` yields `[BOS] -> the`, `[BOS, the] -> film`, `[BOS, the, film] -> was`, and so on. The default stride samples every fourth target to reduce nearly identical overlapping examples. Use `--stride 1` to predict at every position, at increased training cost.

The split happens before vocabulary fitting and window construction. This prevents overlapping windows from a single review appearing in different partitions. `results/split_review_ids.json` records the exact membership, and `results/oov.csv` reports the unknown-token rate.

## Part 2: Models and shared training conditions

| Architecture | Context representation |
|---|---|
| Simple RNN | One tanh recurrent layer; final hidden state |
| Bidirectional RNN | Forward and reverse tanh RNNs; concatenated final hidden states |
| LSTM | Gated hidden and cell states; final hidden state |
| GRU | Update/reset gates; final hidden state |

All models use a learned 64-dimensional embedding, one recurrent layer with 64 hidden units per direction, and a linear vocabulary projection. No pretrained embeddings are used. Training uses cross-entropy on logits, Adam with learning rate 0.003, batch size 256, gradient clipping at norm 1.0, and four epochs. All architectures receive the same data, shuffled batch sequence, vocabulary, initialization seed, hardware, and checkpoint-selection rule. The checkpoint with the lowest validation loss is restored for final evaluation and generation. No early stopping gives every model the same epoch budget.

**Bidirectional correctness:** the reverse RNN reads only the observed input window. It never receives the next-token target or later review text. This is a valid autoregressive conditional model of the next token given a complete past window. Concatenating the two final hidden states summarizes both directions correctly; taking only the last sequence output would not summarize the entire reverse context. Both directions are recomputed after every generated token.

Equal hidden widths do not mean equal parameter counts. Gated models and the bidirectional model have different capacities; the measured table reports parameter counts. This is a controlled practical comparison, not a parameter-matched architecture study.

## Execution

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python autoregressive_imdb.py --epochs 4 --output results_reproduced
```

The first run downloads the roughly 84 MB original archive into `data/`. The script reads reviews directly from the archive without extracting it. CPU is the default and uses four computation threads. Optional GPU run: `python autoregressive_imdb.py --device cuda --output results_gpu`. Run on the same device for all four architectures. The supplied measurements are in `results/`; the command above writes a fresh run to `results_reproduced`. An existing nonempty output folder is rejected to preserve previous results.

Alternatively, open `autoregressive_imdb.ipynb` in Jupyter or Colab after installing the dependencies, and run its cells in order. It writes to `results_notebook` and includes the entire implementation. The notebook is supplied with empty execution outputs; the completed measured experiment is in `results/`.

For a small pipeline check: `python autoregressive_imdb.py --reviews 100 --epochs 1 --output smoke_results`. A smoke run is not evidence for the full assignment results. For a larger experiment: `python autoregressive_imdb.py --reviews 10000 --max-tokens 100 --epochs 10 --output results_larger`.

## Part 3: Quantitative evaluation and measured results

Each epoch records training/validation cross-entropy, perplexity, top-1 next-token accuracy, and duration. Cross-entropy is averaged by the number of target examples, not by the number of batches. With natural logarithms, `perplexity = exp(mean_cross_entropy)`. Accuracy is the fraction of targets whose IDs equal the highest-logit prediction. Lower loss/perplexity and higher accuracy indicate better prediction under the same tokenization and vocabulary.

Timing covers the training loop including batch loading, forward/backward passes, gradient clipping, and optimizer updates. It excludes validation, checkpoint writing, dataset preparation, and generation. GPU timing is synchronized. The reported value is mean training seconds per epoch; individual epochs are saved separately.

<!-- MEASURED_RESULTS_START -->
| Model | Parameters | Selected epoch | Train loss | Validation loss | Validation perplexity | Validation accuracy | Test perplexity | Training seconds/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RNN | 524320 | 4 | 4.6389 | 4.9772 | 145.07 | 15.46% | 141.73 | 7.39 |
| Bidirectional RNN | 788640 | 2 | 4.9483 | 5.0497 | 155.97 | 14.80% | 151.84 | 12.37 |
| LSTM | 549280 | 4 | 4.7744 | 4.9718 | 144.29 | 15.73% | 142.62 | 10.11 |
| GRU | 540960 | 4 | 4.6362 | 4.9759 | 144.88 | 15.35% | 141.32 | 13.40 |

These are actual measurements from the completed four-epoch CPU run (Python 3.12.14, PyTorch 2.14.1+cpu, Windows 11, four computation threads). There were 82,842 training, 10,368 validation, and 10,289 test windows. Unknown-token rates were 8.83%, 10.14%, and 9.86%, respectively.

**Comparison:** LSTM achieved the lowest validation loss/perplexity and highest validation accuracy. GRU had the lowest test perplexity; LSTM had the highest test accuracy (16.05%). Simple RNN was fastest. The bidirectional model overfit after epoch 2: its training loss continued falling while validation loss rose. All other selected checkpoints came from epoch 4.
<!-- MEASURED_RESULTS_END -->

See [the complete measured report](results/experiment_report.md), [epoch history](results/history.csv), and [learning curves](results/learning_curves.png). Test loss and accuracy are also included in `summary.csv`. The lowest validation loss selects the quantitative winner; test scores describe the held-out performance of the selected checkpoints.

Perplexity is measured on stride-sampled targets with UNK included, not on every word of every full review. A small vocabulary can artificially simplify rare-word prediction, so unknown rates must be considered. Scores are comparable within this experiment and should not be compared directly with models using different vocabularies or tokenization.

## Part 4: Generation and sampling-temperature experiment

At each step the model receives the last 16 observed/generated tokens. It samples from `softmax(logits / temperature)`, appends that token, and repeats until EOS or 50 new tokens. PAD, BOS, and UNK are masked only during generation to avoid displaying special tokens. Evaluation uses the unmasked model distribution. There is no top-k/top-p filter or repetition penalty, so temperature effects are directly observable.

The shared seeds are `the movie was`, `the story and the characters`, and `i really enjoyed this film because`. Every model uses temperatures 0.2, 0.7, and 1.2 and three random replicates per seed/temperature, giving 108 continuations. Corresponding comparisons use the same random seeds; model-specific probability distributions still produce different texts.

| Temperature | Expected distribution effect | Quality trade-off to examine |
|---|---|---|
| 0.2 | Sharply concentrated on likely tokens | Predictable phrases; often repetitive and limited in diversity |
| 0.7 | Moderately concentrated | Possible balance between recognizable local phrases and variation |
| 1.2 | Flatter, allowing more unlikely tokens | Greater lexical variety; more risk of incoherence and topic drift |

These are expectations, not substitutes for the measured samples. See [generated continuations](results/samples.csv) and [temperature statistics](results/temperature_summary.csv). The report includes the first replicate for each model/seed/temperature combination.

Distinct-1 and distinct-2 are unique unigrams/bigrams divided by the corresponding total count **within each continuation**. Repeated-bigram fraction is `1 - distinct_2`. Tables report the mean across the nine samples for each model/temperature. Undefined statistics for empty/one-token outputs are blank rather than treated as perfect diversity. Output length is reported because early EOS and short samples can inflate distinctness.

### Qualitative assessment

Read continuations together with their seeds and judge:

- **Coherence:** grammar, understandable local phrases, and sentence flow.
- **Contextual consistency:** whether the continuation stays relevant to the seed, movie reviewing, and its developing opinion.
- **Diversity:** variation of words, constructions, and ideas, using lexical statistics as supporting evidence.
- **Repetition:** repeated tokens, phrases, or semantic loops.

`qualitative_ratings.csv` provides a worksheet. Rate coherence and context from 1 (poor) to 5 (strong), and repetition from 1 (none) to 5 (severe), with notes. Automated lexical statistics cannot establish semantic quality. Assessments below are an informal reading of samples, not a blinded human study.

<!-- QUALITATIVE_ANALYSIS_START -->
### Observed temperature effects

For the selected LSTM, averages over nine outputs per temperature were:

| Temperature | Mean new tokens | Distinct-1 | Distinct-2 | Repeated bigram fraction |
|---|---:|---:|---:|---:|
| 0.2 | 2.67 | 0.964 | 1.000 | 0.000 |
| 0.7 | 12.11 | 0.891 | 0.990 | 0.010 |
| 1.2 | 30.22 | 0.919 | 0.997 | 0.003 |

At **0.2**, the LSTM often produced very short fragments. For `i really enjoyed this film because`, replicate 0 produced `it is a`; for `the movie was`, it emitted EOS immediately. Its apparently high diversity and zero repeated bigrams therefore do not indicate successful long-form generation. The GRU and bidirectional RNN produced longer, more repetitive outputs: their mean repeated-bigram fractions were 0.260 and 0.273. A GRU sample repeatedly reused `i was a lot of the movie`.

At **0.7**, outputs showed less looping than the low-temperature GRU/bidirectional outputs, with occasional recognizable movie-review phrases. A bidirectional sample following `the story and the characters` continued `, the acting is awful . it is a different`, which is locally readable but unfinished. Context consistency was unreliable: that model also continued the positive seed with `it . i have watched it all to say that it was`, which does not develop the seed's stated reason. The GRU sometimes sustained longer outputs (mean 22.78 new tokens) but still had grammatical and opinion-consistency problems. Temperature 0.7 is a reasonable provisional compromise, not a demonstrated guarantee of coherence.

At **1.2**, almost all models produced approximately 29-30 new tokens per sample and nearly unique bigrams. They also drifted into loosely connected words and names. For example, the GRU continued `the movie was` with `bob to my own , night british up`. Greater lexical diversity did not translate into meaningful or contextually consistent sentences. Higher temperature did not strictly increase distinct-1 for every model, partly because output lengths changed.

### Qualitative model comparison

| Model | Coherence and context | Diversity and repetition |
|---|---|---|
| Simple RNN | Recognizable short review phrases, with incomplete grammar; immediate EOS for one shared seed in replicate 0 at all temperatures | Low-temperature repetition in some samples; high temperature becomes word salad |
| Bidirectional RNN | Some locally readable review phrases at 0.7, but weak global continuity | Most repeated bigrams at 0.2; less looping at higher temperatures |
| LSTM | Best validation prediction scores, but short low-temperature fragments and unreliable extended grammar | Low apparent repetition is partly explained by short outputs; 1.2 increases length without fixing coherence |
| GRU | Longer continuations at 0.7; recognizable review language mixed with grammar errors and opinion drift | Strong loops at 0.2; more varied but incoherent at 1.2 |

These observations come from reading the representative replicate-0 outputs across all seeds and temperatures. The supplied rating worksheet remains blank; no numerical human ratings were fabricated. Among all 108 samples, five stopped immediately. Complete continuations and all replicate statistics are saved for inspection.

## Final conclusion

**LSTM is the selected best model under the predefined validation-loss criterion**, with validation perplexity 144.29 and validation accuracy 15.73%. It narrowly outperforms GRU and Simple RNN on that criterion and trains faster than GRU and the bidirectional RNN in this run. GRU has the lowest held-out test perplexity (141.32), while Simple RNN offers the strongest speed advantage (7.39 seconds per epoch). Consequently, these data support a tentative choice of LSTM for predictive performance, not a decisive claim that it generates the most coherent text.

For generation, 0.7 is the most defensible starting temperature given the observed loops at 0.2 and incoherence at 1.2. None of the four models consistently generates fluent, sustained reviews after only four epochs on truncated data. Longer training with validation-based stopping, a larger or better-handled vocabulary, full-coverage sliding windows, and repeated training seeds are needed before making a strong quality claim. Repetition penalties or top-p sampling could be studied separately, but were intentionally absent from this temperature-only comparison.
<!-- QUALITATIVE_ANALYSIS_END -->

## Limitations and reproducibility

This is a single-seed, short-training experiment on truncated review prefixes. It cannot establish universal architecture superiority, long-document quality, or reliable differences between close scores. Repeated training seeds and longer training would strengthen conclusions. Shared seed and batch order aid reproducibility, but numerical results and timing can vary by PyTorch version and hardware. `config.json` records software, hardware description, hyperparameters, archive SHA-256, and sample counts. Sampling statistics use three replicates, but this does not measure variability across independently trained models.

## Repository contents and submission

```text
autoregressive_imdb.py       Complete commented implementation (Parts 1-4)
autoregressive_imdb.ipynb    Modular notebook containing the same implementation
requirements.txt            Python dependencies
README.md                   Problem, method, execution, results, discussion
results/
  config.json               Experiment and environment metadata
  split_review_ids.json     Exact review-level split membership
  vocab.json                Training-only token-to-ID vocabulary
  oov.csv                   Unknown-token statistics
  history.csv               Per-epoch loss, perplexity, accuracy and duration
  summary.csv               Selected checkpoint metrics and parameter counts
  learning_curves.png       Training and validation plots
  samples.csv               All 108 generation outputs
  temperature_summary.csv   Per-temperature lexical statistics
  qualitative_ratings.csv   Optional human-rating worksheet
  experiment_report.md      Measured tables and representative continuations
  checkpoints/              Locally saved weights (excluded by .gitignore)
```

Upload the script, dependencies, README, results, plots, and generated samples to a clearly named GitHub repository. Raw data and local dependency files are excluded. If required, add model weights explicitly or attach them as release assets. For a private repository, provide evaluator access.

The assignment also requires a Word document named `YourName_RegNum_FGAI_Assignment2.docx`, containing student name, registration number, assignment title, system title, dataset name/link, and GitHub repository link. Convert that document to PDF and submit it to the specified LMS section. The repository is published at [Noob-ooN/fgai-autoregressive-imdb-rnn](https://github.com/Noob-ooN/fgai-autoregressive-imdb-rnn). A student-details PDF with editable name and registration-number placeholders is supplied as `YourName_RegNum_FGAI_Assignment2.pdf`; fill these fields and rename the document before submitting to LMS.

## Implementation references

- [PyTorch recurrent layers](https://docs.pytorch.org/docs/stable/nn.html): RNN, LSTM, GRU, embedding, and linear layers.
- [PyTorch cross-entropy loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): logits and integer class targets.
