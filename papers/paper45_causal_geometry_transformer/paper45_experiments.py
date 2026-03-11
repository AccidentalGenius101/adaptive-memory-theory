"""
Paper 45: Transformer Causal Geometry Probe
============================================
Question: Does intervention-defined influence predict representation geometry
          better than sequence distance alone?

Design:
  Model:    GPT-2 small (12 transformer layers, d=768)
  Dataset:  50 sentences from UD English EWT with dependency parses
  Ablation: mean embedding (main) + random token (control)
  Metrics:  seq_dist, attention rollout, ablation influence, geom distance
  Analysis: layer-wise partial correlations, mixed-effects regression,
            dependency-distance matched comparison

Core prediction:
  After controlling for sequence distance, ablation influence should predict
  representation geometry. Effect should peak in middle-to-late layers.

Run: py paper45_experiments.py
"""

import os, json, sys, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────

N_SENTENCES  = 50
MIN_SEQ_LEN  = 6
MAX_SEQ_LEN  = 35   # GPT-2 subtokens; skip long sentences (slow ablation)
RESULTS_FILE = "paper45_results.json"
ANALYSIS_FILE= "paper45_analysis.json"
RANDOM_SEED  = 42

# ── Dependency checks ─────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ["transformers", "datasets", "torch", "pandas", "scipy", "statsmodels", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import torch
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import statsmodels.formula.api as smf
from transformers import GPT2Model, GPT2Tokenizer

# ── Load model ────────────────────────────────────────────────────────────────

def load_gpt2():
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # attn_implementation='eager' required to get attention weights back
    # (newer transformers default to 'sdpa' which doesn't return them)
    model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    with torch.no_grad():
        mean_embed = model.wte.weight.mean(0).detach()

    n_layers = model.config.n_layer
    print(f"  n_layers={n_layers}, d_model={model.config.n_embd}, vocab={model.config.vocab_size}")
    return tokenizer, model, mean_embed

# ── Word-to-subtoken alignment ────────────────────────────────────────────────

def word_token_spans(words, tokenizer):
    """
    Map each UD word to (start_subtoken, end_subtoken) in GPT-2 tokenization.
    Returns None if alignment fails.
    """
    spans = []
    idx = 0
    for i, word in enumerate(words):
        prefix = " " if i > 0 else ""
        toks = tokenizer.encode(prefix + word, add_special_tokens=False)
        if len(toks) == 0:
            return None
        spans.append((idx, idx + len(toks)))
        idx += len(toks)
    return spans

# ── Load UD sentences ─────────────────────────────────────────────────────────

UD_EWT_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_English-EWT/master/en_ewt-ud-train.conllu"
)
UD_LOCAL   = "en_ewt-ud-train.conllu"


def _download_ud():
    """Download UD English EWT train split if not already present."""
    import urllib.request
    if os.path.exists(UD_LOCAL):
        print(f"  Using cached {UD_LOCAL}")
        return
    print(f"  Downloading UD English EWT from GitHub...")
    urllib.request.urlretrieve(UD_EWT_URL, UD_LOCAL)
    print(f"  Saved to {UD_LOCAL}")


def load_ud_sentences(tokenizer, n=N_SENTENCES):
    import conllu as conllu_lib

    print(f"Loading {n} UD English EWT sentences...")
    _download_ud()

    with open(UD_LOCAL, encoding="utf-8") as f:
        raw = f.read()

    all_sents = conllu_lib.parse(raw)
    sentences = []

    for sent in all_sents:
        # Keep only integer-indexed tokens (skip multiword / empty nodes)
        tokens  = [t for t in sent if isinstance(t["id"], int)]
        words   = [t["form"]   for t in tokens]
        heads   = [t["head"]   for t in tokens]   # 1-indexed; 0 = root
        deprels = [t["deprel"] for t in tokens]

        # Tokenize full sentence
        input_ids = tokenizer.encode(" ".join(words), return_tensors="pt")
        seq_len   = input_ids.shape[1]

        if not (MIN_SEQ_LEN <= seq_len <= MAX_SEQ_LEN):
            continue

        spans = word_token_spans(words, tokenizer)
        if spans is None or len(spans) != len(words):
            continue
        if spans[-1][1] != seq_len:
            continue

        # Build dependency pairs at first-subtoken level
        dep_pairs = set()
        for w_idx, (head_idx, deprel) in enumerate(zip(heads, deprels)):
            if not head_idx or head_idx == 0:
                continue
            dep_tok  = spans[w_idx][0]
            head_tok = spans[head_idx - 1][0]
            if dep_tok == head_tok:
                continue
            a, b = min(dep_tok, head_tok), max(dep_tok, head_tok)
            dep_pairs.add((a, b))

        sentences.append({
            "words":     words,
            "input_ids": input_ids,
            "seq_len":   seq_len,
            "dep_pairs": list(dep_pairs),
        })

        if len(sentences) >= n:
            break

    print(f"  Collected {len(sentences)} usable sentences")
    return sentences

# ── Forward passes ────────────────────────────────────────────────────────────

def clean_forward(model, input_ids):
    """
    Returns:
      hidden: np.array [n_layers+1, seq_len, d]  (layer 0 = embedding)
      rollout: np.array [seq_len, seq_len]         rollout[i,j] = j's influence on i
    """
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, output_attentions=True)

    hidden = np.stack(
        [h[0].detach().cpu().numpy() for h in out.hidden_states], axis=0
    )  # [n_layers+1, seq_len, d]

    rollout = _attention_rollout(out.attentions, input_ids.shape[1])
    return hidden, rollout


def _attention_rollout(attentions, seq_len):
    """
    Accumulate attention flow across layers with residual connection.
    Returns [seq_len, seq_len]: rollout[i, j] = total influence of j on i.
    Falls back to uniform (1/seq_len) if attentions are unavailable.
    """
    if attentions is None or any(a is None for a in attentions):
        return np.full((seq_len, seq_len), 1.0 / seq_len)

    R = torch.eye(seq_len)
    for layer_attn in attentions:
        # layer_attn: [1, n_heads, seq_len, seq_len]
        avg = layer_attn[0].mean(0)               # [seq_len, seq_len]
        aug = 0.5 * avg + 0.5 * torch.eye(seq_len)
        aug = aug / aug.sum(-1, keepdim=True)
        R   = aug @ R
    return R.detach().numpy()


def ablated_hidden(model, input_ids, j, replacement_embed):
    """
    Replace token j's word embedding with replacement_embed (no positional change).
    Returns hidden: np.array [n_layers+1, seq_len, d].
    """
    with torch.no_grad():
        word_embeds = model.wte(input_ids).clone()
        word_embeds[0, j] = replacement_embed
        out = model(inputs_embeds=word_embeds, output_hidden_states=True)

    return np.stack(
        [h[0].detach().cpu().numpy() for h in out.hidden_states], axis=0
    )

# ── Per-sentence processing ───────────────────────────────────────────────────

def process_sentence(s_idx, sent, model, mean_embed, rand_embed):
    """
    For all directed pairs (j -> i) with j < i, compute:
      - seq_dist, is_dep
      - attn_rollout(j->i)
      - ablation_inf(j->i) per layer   [mean ablation]
      - rand_inf(j->i) per layer       [random-token ablation, control]
      - geom_dist(i, j) per layer

    Returns list of dicts (one per pair).
    """
    input_ids = sent["input_ids"]
    seq_len   = sent["seq_len"]
    dep_set   = {tuple(p) for p in sent["dep_pairs"]}

    hidden_clean, rollout = clean_forward(model, input_ids)
    n_layers = hidden_clean.shape[0]

    # Pre-compute ablation hidden states for each source j
    abl_hidden  = {}   # j -> [n_layers, seq_len, d]
    rand_hidden = {}

    for j in range(seq_len - 1):
        abl_hidden[j]  = ablated_hidden(model, input_ids, j, mean_embed)
        rand_hidden[j] = ablated_hidden(model, input_ids, j, rand_embed)

    # Build records for all (j, i) pairs with j < i
    records = []
    for j in range(seq_len - 1):
        for i in range(j + 1, seq_len):
            is_dep = 1 if (j, i) in dep_set else 0

            # Geometric distance per layer: 1 - cos(h_i, h_j)
            geom = []
            for l in range(n_layers):
                hi = hidden_clean[l, i]
                hj = hidden_clean[l, j]
                cos = np.dot(hi, hj) / (np.linalg.norm(hi) * np.linalg.norm(hj) + 1e-8)
                geom.append(float(1.0 - cos))

            # Ablation influence: ||h_i_clean - h_i_ablated|| per layer
            abl_inf  = [float(np.linalg.norm(hidden_clean[l, i] - abl_hidden[j][l, i]))
                        for l in range(n_layers)]
            rand_inf = [float(np.linalg.norm(hidden_clean[l, i] - rand_hidden[j][l, i]))
                        for l in range(n_layers)]

            records.append({
                "sent_idx":     s_idx,
                "i":            i,
                "j":            j,
                "seq_dist":     i - j,
                "seq_dist_n":   (i - j) / seq_len,
                "is_dep":       is_dep,
                "attn_rollout": float(rollout[i, j]),
                "geom":         geom,       # [n_layers]
                "abl_inf":      abl_inf,    # [n_layers]
                "rand_inf":     rand_inf,   # [n_layers]
            })

    return records

# ── Analysis ──────────────────────────────────────────────────────────────────

def partial_corr(y, x1, x2):
    """
    Partial correlation of y with x1, controlling for x2.
    Uses OLS residualisation: regress both y and x1 on x2, correlate residuals.
    """
    X = np.column_stack([np.ones_like(x2), x2])
    def resid(v):
        coef = np.linalg.lstsq(X, v, rcond=None)[0]
        return v - X @ coef
    r, p = pearsonr(resid(y), resid(x1))
    return float(r), float(p)


def analyze(records, n_layers):
    df = pd.DataFrame(records)

    # Expand list columns into per-layer scalar columns
    for l in range(n_layers):
        df[f"geom_{l}"]    = df["geom"].apply(lambda x: x[l])
        df[f"abl_inf_{l}"] = df["abl_inf"].apply(lambda x: x[l])
        df[f"rand_inf_{l}"]= df["rand_inf"].apply(lambda x: x[l])

    df = df.drop(columns=["geom", "abl_inf", "rand_inf"])
    df = df.dropna()

    n_dep    = df["is_dep"].sum()
    n_nondep = (df["is_dep"] == 0).sum()
    print(f"\n  Total pairs: {len(df)}  |  dep: {n_dep}  non-dep: {n_nondep}")

    # ── Layer-wise partial correlations ───────────────────────────────────────
    print("\n=== Layer-wise Partial Correlations ===")
    print(f"{'L':>3}  {'pcorr(geom,abl|seq)':>20}  {'p_abl':>8}  "
          f"{'pcorr(geom,roll|seq)':>21}  {'p_roll':>8}  "
          f"{'pcorr(geom,rand|seq)':>21}  {'p_rand':>8}")

    layer_stats = []
    for l in range(n_layers):
        g   = df[f"geom_{l}"].values
        ai  = df[f"abl_inf_{l}"].values
        ri  = df[f"rand_inf_{l}"].values
        rol = df["attn_rollout"].values
        sd  = df["seq_dist_n"].values

        r_abl,  p_abl  = partial_corr(g, ai,  sd)
        r_rand, p_rand = partial_corr(g, ri,  sd)
        r_roll, p_roll = partial_corr(g, rol, sd)

        def sig(p):
            return "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "

        print(f"{l:>3}  {r_abl:>+.4f}{sig(p_abl)}  {p_abl:>8.4f}  "
              f"{r_roll:>+.4f}{sig(p_roll)}  {p_roll:>8.4f}  "
              f"{r_rand:>+.4f}{sig(p_rand)}  {p_rand:>8.4f}")

        layer_stats.append({
            "layer": l,
            "pcorr_abl":  r_abl,  "p_abl":  p_abl,
            "pcorr_roll": r_roll, "p_roll": p_roll,
            "pcorr_rand": r_rand, "p_rand": p_rand,
        })

    # ── Mixed-effects regression at two representative layers ─────────────────
    print("\n=== Mixed-Effects Regression ===")
    target_layers = [n_layers // 2, n_layers - 1]    # mid and last

    me_results = {}
    for l in target_layers:
        sub = df[["sent_idx", "seq_dist_n", f"abl_inf_{l}",
                  f"geom_{l}", "is_dep"]].copy()
        sub.columns = ["sent_idx", "seq_dist", "abl_inf", "geom", "is_dep"]
        sub = sub.dropna()

        try:
            m0 = smf.mixedlm("geom ~ seq_dist",
                              sub, groups=sub["sent_idx"]).fit(reml=False, method="lbfgs",
                                                               disp=False)
            m1 = smf.mixedlm("geom ~ seq_dist + abl_inf + is_dep",
                              sub, groups=sub["sent_idx"]).fit(reml=False, method="lbfgs",
                                                               disp=False)

            d_aic = m1.aic - m0.aic
            c_abl = m1.params.get("abl_inf", float("nan"))
            p_abl = m1.pvalues.get("abl_inf", float("nan"))
            c_dep = m1.params.get("is_dep",  float("nan"))
            p_dep = m1.pvalues.get("is_dep",  float("nan"))

            print(f"  Layer {l:>2}: AIC_base={m0.aic:.1f}  AIC_full={m1.aic:.1f}  "
                  f"dAIC={d_aic:+.1f}  "
                  f"abl_inf beta={c_abl:+.4f} (p={p_abl:.4f})  "
                  f"is_dep  beta={c_dep:+.4f} (p={p_dep:.4f})")

            me_results[l] = {"dAIC": d_aic, "abl_coef": c_abl, "p_abl": p_abl,
                             "dep_coef": c_dep, "p_dep": p_dep}
        except Exception as e:
            print(f"  Layer {l}: regression failed ({e})")

    # ── Dependency-distance matched analysis (at layer n_layers//2) ───────────
    L_rep = n_layers // 2
    print(f"\n=== Dependency-Matched Analysis (layer {L_rep}) ===")
    print(f"{'Seq dist':>10}  {'dep geom':>10}  {'non-dep geom':>13}  {'t':>6}  {'p':>6}  {'n_dep':>6}")

    matched_results = []
    for lo, hi in [(1, 3), (3, 6), (6, 15)]:
        band = df[(df["seq_dist"] >= lo) & (df["seq_dist"] < hi)]
        dep_g  = band[band["is_dep"] == 1][f"geom_{L_rep}"].values
        ndep_g = band[band["is_dep"] == 0][f"geom_{L_rep}"].values
        if len(dep_g) > 2 and len(ndep_g) > 2:
            t, p = ttest_ind(dep_g, ndep_g)
            print(f"  [{lo:2d},{hi:2d})  dep={dep_g.mean():.3f}+-{dep_g.std():.3f}  "
                  f"ndep={ndep_g.mean():.3f}+-{ndep_g.std():.3f}  "
                  f"t={t:+.2f}  p={p:.3f}  n_dep={len(dep_g)}")
            matched_results.append({"range": f"[{lo},{hi})", "dep_mean": float(dep_g.mean()),
                                     "ndep_mean": float(ndep_g.mean()), "t": float(t), "p": float(p)})
        else:
            print(f"  [{lo:2d},{hi:2d})  insufficient dep pairs (n={len(dep_g)})")

    # ── Ablation vs rollout comparison (best layer) ───────────────────────────
    best_l = max(layer_stats, key=lambda s: abs(s["pcorr_abl"]))["layer"]
    print(f"\n=== Best layer: {best_l} ===")
    ls = layer_stats[best_l]
    print(f"  pcorr(geom, abl_inf | seq):  {ls['pcorr_abl']:+.4f}  p={ls['p_abl']:.4f}")
    print(f"  pcorr(geom, rollout | seq):  {ls['pcorr_roll']:+.4f}  p={ls['p_roll']:.4f}")
    print(f"  pcorr(geom, rand_inf | seq): {ls['pcorr_rand']:+.4f}  p={ls['p_rand']:.4f}")
    print(f"  (rand is ablation control -- should be weaker than abl if result is real)")

    return {
        "n_sentences": int(df["sent_idx"].nunique()),
        "n_pairs":     int(len(df)),
        "n_dep":       int(n_dep),
        "n_layers":    n_layers,
        "layer_stats": layer_stats,
        "me_results":  me_results,
        "matched":     matched_results,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    tokenizer, model, mean_embed = load_gpt2()
    n_layers = model.config.n_layer + 1   # +1 for embedding layer (layer 0)

    # Fixed random token for control ablation (same across sentences for reproducibility)
    rand_tok   = int(rng.integers(200, model.wte.weight.shape[0]))
    rand_embed = model.wte.weight[rand_tok].detach()
    print(f"  Random control token id: {rand_tok}")

    # Load or resume results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            saved = json.load(f)
        all_records = saved["records"]
        done_sents  = set(saved["done_sents"])
        print(f"Resuming: {len(done_sents)} sentences already processed "
              f"({len(all_records)} pairs)")
    else:
        all_records = []
        done_sents  = set()

    sentences = load_ud_sentences(tokenizer, n=N_SENTENCES)

    print(f"\nProcessing {len(sentences)} sentences...")
    print(f"  (Each sentence requires seq_len ablation passes -> may take a few minutes)")

    for s_idx, sent in enumerate(sentences):
        if s_idx in done_sents:
            continue

        t0 = time.time()
        records = process_sentence(s_idx, sent, model, mean_embed, rand_embed)
        all_records.extend(records)
        done_sents.add(s_idx)

        n_dep = sum(1 for r in records if r["is_dep"])
        print(f"  [{s_idx+1:3d}/{len(sentences)}] len={sent['seq_len']:2d}  "
              f"pairs={len(records):3d}  dep={n_dep:2d}  "
              f"{time.time()-t0:.1f}s")

        # Save after each sentence (restart-safe)
        with open(RESULTS_FILE, "w") as f:
            json.dump({"records": all_records,
                       "done_sents": list(done_sents)}, f)

    print(f"\nCollection done: {len(done_sents)} sentences, {len(all_records)} pairs")

    analysis = analyze(all_records, n_layers)

    with open(ANALYSIS_FILE, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nSaved: {RESULTS_FILE}, {ANALYSIS_FILE}")
    print("Next: py paper45_figure1.py")


if __name__ == "__main__":
    main()
