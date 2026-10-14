"""ICT4403 Assignment 2: reproducible word-level recurrent language modelling.

Run: python autoregressive_imdb.py --epochs 4
Only prefixes are passed to any model, including the bidirectional RNN.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import random
import re
import tarfile
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

MODELS = ('Simple RNN', 'Bidirectional RNN', 'LSTM', 'GRU')
SOURCE = 'https://ai.stanford.edu/~amaas/data/sentiment/'
DOWNLOAD = SOURCE + 'aclImdb_v1.tar.gz'
SEEDS = ('the movie was', 'the story and the characters', 'i really enjoyed this film because')
TEMPERATURES = (0.2, 0.7, 1.2)


# Part 1: Data preprocessing, tokenization and autoregressive sequence construction.
def tokenize(text):
    """Preserve contractions and punctuation; remove HTML and normalize case."""
    text = html.unescape(re.sub(r'<[^>]+>', ' ', text)).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;:]", text)


def load_reviews(data_dir, count, seed):
    """Read selected labelled TRAIN reviews directly from tar; never extract files.

    Sentiment labels only balance the subsample, never become prediction targets.
    The official test partition and unsupervised reviews are not used.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / 'aclImdb_v1.tar.gz'
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        print('Downloading the original Stanford IMDb archive...', flush=True)
        urllib.request.urlretrieve(DOWNLOAD, temporary)
        temporary.replace(archive)
    rng = random.Random(seed)
    identities = []
    with tarfile.open(archive, 'r:gz') as tar:
        for sentiment in ('pos', 'neg'):
            members = sorted((m for m in tar.getmembers()
                              if m.isfile() and m.name.startswith(f'aclImdb/train/{sentiment}/')
                              and m.name.endswith('.txt')), key=lambda m: m.name)
            if count // 2 > len(members):
                raise ValueError('Too many requested reviews.')
            for m in rng.sample(members, count // 2):
                identities.append(m.name)
    # A compressed tar is not random-access: reading shuffled members repeatedly
    # would decompress the archive thousands of times. Stream through it once.
    selected = set(identities)
    texts = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name in selected:
                texts[member.name] = tar.extractfile(member).read().decode('utf-8')
    reviews = [texts[name] for name in identities]
    order = list(range(len(reviews)))
    rng.shuffle(order)
    return [reviews[i] for i in order], [identities[i] for i in order], archive


class Windows(Dataset):
    """Fixed length context -> ONE unseen next token, without crossing reviews.

    Start with BOS; predict EOS at the review end when selected by stride.
    Short prefixes are left padded. A stride reduces the overlapping sample count.
    """
    def __init__(self, reviews, vocab, context, max_tokens, stride):
        self.context = context
        self.encoded = [[2] + [vocab.get(w, 1) for w in r[:max_tokens]] + [3]
                        for r in reviews]
        self.index = [(i, j) for i, r in enumerate(self.encoded)
                      for j in range(1, len(r), stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, j = self.index[index]
        prefix = self.encoded[i][max(0, j - self.context):j]
        x = [0] * (self.context - len(prefix)) + prefix
        return torch.tensor(x, dtype=torch.long), torch.tensor(self.encoded[i][j])


# Part 2: Four architectures and common training conditions.
class LanguageModel(nn.Module):
    def __init__(self, architecture, vocab_size, embedding=64, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        cls = {'Simple RNN': nn.RNN, 'Bidirectional RNN': nn.RNN,
               'LSTM': nn.LSTM, 'GRU': nn.GRU}[architecture]
        self.recurrent = cls(embedding, hidden, batch_first=True,
                             bidirectional=architecture == 'Bidirectional RNN')
        self.output = nn.Linear(hidden * (2 if architecture == 'Bidirectional RNN' else 1),
                                vocab_size)

    def forward(self, x):
        _, hidden = self.recurrent(self.embedding(x))
        if isinstance(hidden, tuple):  # LSTM: use hidden state, not cell state.
            hidden = hidden[0]
        # h_n contains the complete forward AND reverse context summaries.
        # output[:, -1] alone would give an incomplete reverse summary.
        summary = hidden.transpose(0, 1).reshape(x.shape[0], -1)
        return self.output(summary)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Part 3: Example-weighted loss/accuracy, perplexity, and synchronized timing.
def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    synchronize(device)
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = y.numel()
            total_loss += loss.item() * n
            correct += (logits.argmax(-1) == y).sum().item()
            count += n
    synchronize(device)
    loss = total_loss / count
    return dict(loss=loss, perplexity=math.exp(loss), accuracy=correct / count,
                seconds=time.perf_counter() - start, examples=count)


# Part 4: Sequential generation and sampling-temperature analysis.
@torch.no_grad()
def generate(model, seed, temperature, vocab, context, device, random_seed, max_new=50):
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    model.eval()
    inverse = list(vocab)
    ids = [2] + [vocab.get(w, 1) for w in tokenize(seed)]
    generator = torch.Generator(device=device).manual_seed(random_seed)
    generated = []
    for _ in range(max_new):
        prefix = ids[-context:]
        x = torch.tensor([[0] * (context - len(prefix)) + prefix], device=device)
        logits = model(x)[0] / temperature
        # Exclude PAD/BOS/UNK from displayed text; retain EOS as a valid stop.
        # Evaluation still uses the full, unchanged model distribution.
        logits[[0, 1, 2]] = -torch.inf
        token = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).item()
        if token == 3:
            break
        ids.append(token)
        generated.append(inverse[token])
    return generated


def diversity(tokens):
    def distinct(n):
        grams = list(zip(*(tokens[i:] for i in range(n))))
        return len(set(grams)) / len(grams) if grams else None
    d1, d2 = distinct(1), distinct(2)
    return dict(length=len(tokens), distinct_1=d1, distinct_2=d2,
                repeated_bigram_fraction=None if d2 is None else 1 - d2)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = dict(zip(MODELS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')))
    for name in MODELS:
        rows = [r for r in history if r['model'] == name]
        epochs = [r['epoch'] for r in rows]
        for ax, metric in zip(axes.flat, ('loss', 'perplexity', 'accuracy', 'seconds')):
            ax.plot(epochs, [r['train_' + metric] for r in rows], label=name + ' train', color=colors[name])
            if metric != 'seconds':
                ax.plot(epochs, [r['val_' + metric] for r in rows], '--', label=name + ' val', color=colors[name])
            ax.set(xlabel='Epoch', ylabel=metric.replace('seconds', 'Training seconds'))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews', type=int, default=5000)
    p.add_argument('--max-tokens', type=int, default=80)
    p.add_argument('--context', type=int, default=16)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--vocab-size', type=int, default=4000)
    p.add_argument('--embedding', type=int, default=64)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('results'))
    a = p.parse_args(argv)
    if (a.reviews < 20 or a.reviews > 25000 or a.reviews % 2 or
        min(a.max_tokens, a.context, a.stride, a.epochs, a.batch_size,
            a.embedding, a.hidden, a.threads) < 1 or a.vocab_size < 5 or a.lr <= 0):
        p.error('Use an even review count in [20,25000], positive sizes/rate and vocab >=5.')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output folder is not empty; select a new --output to preserve results.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'checkpoints').mkdir()
    torch.set_num_threads(a.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if a.device == 'auto' else a.device)
    reviews, identities, archive = load_reviews(a.data_dir, a.reviews, a.seed)
    reviews = [tokenize(r)[:a.max_tokens] for r in reviews]
    n_train, n_val = int(.8 * len(reviews)), int(.1 * len(reviews))
    partitions = (reviews[:n_train], reviews[n_train:n_train+n_val], reviews[n_train+n_val:])
    counts = Counter(w for r in partitions[0] for w in r)
    vocab = {w: i for i, w in enumerate(['<pad>', '<unk>', '<bos>', '<eos>'] +
                                     [w for w, _ in counts.most_common(a.vocab_size - 4)])}
    (a.output / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    datasets = [Windows(r, vocab, a.context, a.max_tokens, a.stride) for r in partitions]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(device_used=str(device), python=platform.python_version(), torch=torch.__version__,
                  platform=platform.platform(), processor=platform.processor(),
                  vocabulary_actual=len(vocab), split_reviews=[len(r) for r in partitions],
                  split_windows=[len(d) for d in datasets], dataset_source=SOURCE,
                  archive_sha256=hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest())
    (a.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    split_ids = dict(train=identities[:n_train], validation=identities[n_train:n_train+n_val],
                     test=identities[n_train+n_val:])
    (a.output / 'split_review_ids.json').write_text(json.dumps(split_ids, indent=2), encoding='utf-8')
    oov = []
    for name, part in zip(('train', 'validation', 'test'), partitions):
        words = [w for r in part for w in r]
        oov.append(dict(split=name, tokens=len(words), unknown_fraction=sum(w not in vocab for w in words)/len(words)))
    write_csv(a.output / 'oov.csv', oov)
    history, summary, samples = [], [], []
    for name in MODELS:
        set_seed(a.seed)
        # Reset shuffling generator so all models see the same batches each epoch.
        loaders = [DataLoader(d, batch_size=a.batch_size, shuffle=i == 0,
                              generator=torch.Generator().manual_seed(a.seed), num_workers=0)
                   for i, d in enumerate(datasets)]
        model = LanguageModel(name, len(vocab), a.embedding, a.hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
        best_loss, best_epoch = math.inf, 0
        checkpoint = a.output / 'checkpoints' / (name.replace(' ', '_') + '.pt')
        for epoch in range(1, a.epochs + 1):
            train = run_epoch(model, loaders[0], device, optimizer)
            val = run_epoch(model, loaders[1], device)
            row = dict(model=name, epoch=epoch)
            row.update({'train_' + k: v for k, v in train.items()})
            row.update({'val_' + k: v for k, v in val.items()})
            history.append(row)
            if val['loss'] < best_loss:
                best_loss, best_epoch = val['loss'], epoch
                torch.save(model.state_dict(), checkpoint)
            write_csv(a.output / 'history.csv', history)
            print(f"{name} epoch {epoch}: train={train['loss']:.4f}, val={val['loss']:.4f}, "
                  f"ppl={val['perplexity']:.2f}, train seconds={train['seconds']:.1f}", flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        val = run_epoch(model, loaders[1], device)
        test = run_epoch(model, loaders[2], device)
        rows = [r for r in history if r['model'] == name]
        result = dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                      best_epoch=best_epoch, best_train_loss=rows[best_epoch-1]['train_loss'],
                      val_loss=val['loss'], val_perplexity=val['perplexity'], val_accuracy=val['accuracy'],
                      test_loss=test['loss'], test_perplexity=test['perplexity'], test_accuracy=test['accuracy'],
                      mean_train_seconds=np.mean([r['train_seconds'] for r in rows]))
        summary.append(result)
        write_csv(a.output / 'summary.csv', summary)
        for seed_id, seed in enumerate(SEEDS):
            for temp in TEMPERATURES:
                for replicate in range(3):
                    sampling_seed = a.seed + 1000 * seed_id + replicate
                    words = generate(model, seed, temp, vocab, a.context, device, sampling_seed)
                    sample = dict(model=name, seed=seed, temperature=temp, replicate=replicate,
                                  sampling_seed=sampling_seed, continuation=' '.join(words), **diversity(words))
                    samples.append(sample)
        write_csv(a.output / 'samples.csv', samples)
    plot_history(history, a.output)
    temperature_rows = []
    for name in MODELS:
        for temp in TEMPERATURES:
            group = [s for s in samples if s['model'] == name and s['temperature'] == temp]
            row = dict(model=name, temperature=temp, samples=len(group))
            for metric in ('length', 'distinct_1', 'distinct_2', 'repeated_bigram_fraction'):
                values = [s[metric] for s in group if s[metric] is not None]
                row[metric] = float(np.mean(values)) if values else None
            temperature_rows.append(row)
    write_csv(a.output / 'temperature_summary.csv', temperature_rows)
    # Human ratings are deliberately blank: lexical statistics cannot measure coherence.
    write_csv(a.output / 'qualitative_ratings.csv', [dict(model=s['model'], seed=s['seed'],
              temperature=s['temperature'], replicate=s['replicate'], coherence_1_to_5='',
              context_1_to_5='', repetition_1_to_5='', notes='') for s in samples])
    winner = min(summary, key=lambda r: r['val_loss'])
    report = ['# Measured experiment report', '',
              f"Selected by lowest validation loss: **{winner['model']}**. Test data was not used for selection.", '',
              '| Model | Parameters | Best epoch | Train loss | Val loss | Val PPL | Val accuracy | Test PPL | Train s/epoch |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['model']} | {r['parameters']} | {r['best_epoch']} | {r['best_train_loss']:.4f} | "
                      f"{r['val_loss']:.4f} | {r['val_perplexity']:.2f} | {r['val_accuracy']:.2%} | "
                      f"{r['test_perplexity']:.2f} | {r['mean_train_seconds']:.2f} |")
    report += ['', '## Temperature statistics', '',
               '| Model | Temperature | Mean length | Distinct-1 | Distinct-2 | Repeated bigrams |',
               '|---|---:|---:|---:|---:|---:|']
    for r in temperature_rows:
        report.append('| ' + ' | '.join(str(r[k]) if k in ('model', 'temperature') else
                      ('n/a' if r[k] is None else f'{r[k]:.3f}') for k in
                      ('model', 'temperature', 'length', 'distinct_1', 'distinct_2', 'repeated_bigram_fraction')) + ' |')
    report += ['', '## Generated samples (replicate 0)', '']
    for s in samples:
        if s['replicate'] == 0:
            report += [f"### {s['model']}, T={s['temperature']}, seed: {s['seed']}", '', s['continuation'] or '(EOS immediately)', '']
    (a.output / 'experiment_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f"Completed. Measured results: {a.output.resolve()}", flush=True)


if __name__ == '__main__':
    main()
