##############################################
# AI Ouroboros Simulation (Perplexity Fixed + Smoothed d2)
# - Dedicated eval model (loaded once)
# - Sliding-window perplexity on pooled text (stable)
# - Multi-model synthetic generation + fine α sweep
# - Distinct-2 smoothed for noise reduction
##############################################

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, numpy as np, random, math, gc, time
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
from scipy.ndimage import uniform_filter1d
import pandas as pd

# ------------------------
# 0) SETTINGS (tune if you want)
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Generation models (create synthetic text from these)
gen_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-1_5",
    "distilgpt2"
]

# Dedicated evaluation model for perplexity
eval_model_name = "gpt2-large" if device == "cuda" else "gpt2-medium"

# Experiment parameters
synthetic_samples_per_model = 400
alphas = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]
prompt_template = "Write a short news headline about technology:\n"

# Sliding-window perplexity settings
max_length = 1024
stride = 768

# Dataset
human_dataset = "ag_news"
human_subset = 1000

# Deterministic-ish generation
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# ------------------------
# 1) load human texts
# ------------------------
dataset = load_dataset(human_dataset, split=f"train[:{human_subset}]")
human_texts = [ex.get("text", ex.get("title", "")) for ex in dataset]
print(f"Loaded {len(human_texts)} human samples")

# ------------------------
# 2) helper functions (ngram, kl, distinct)
# ------------------------
def ngram_dist(texts, n=2):
    ngrams = Counter()
    for t in texts:
        tokens = t.split()
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i:i+n])] += 1
    total = sum(ngrams.values())
    return {k: v/total for k, v in ngrams.items()} if total > 0 else {}

def kl_divergence(p_counts, q_counts):
    keys = set(p_counts.keys()) | set(q_counts.keys())
    p = np.array([p_counts.get(k, 1e-12) for k in keys])
    q = np.array([q_counts.get(k, 1e-12) for k in keys])
    return entropy(p, q)

def distinct_n(texts, n=2):
    ngrams = []
    for t in texts:
        tokens = t.split()
        for i in range(len(tokens)-n+1):
            ngrams.append(tuple(tokens[i:i+n]))
    return len(set(ngrams)) / max(len(ngrams), 1)

# ------------------------
# 3) synthetic generation
# ------------------------
def generate_synthetic_for_model(model_name, samples):
    print(f"\nGenerating {samples} samples with {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    out = []
    for _ in tqdm(range(samples)):
        inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        out.append(text.strip())

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out

synthetic_pools = {}
start = time.time()
for gm in gen_models:
    synthetic_pools[gm] = generate_synthetic_for_model(gm, synthetic_samples_per_model)
print(f"\nCompleted generation in {(time.time()-start)/60:.2f} minutes")

# ------------------------
# 4) evaluation model
# ------------------------
print(f"\nLoading evaluation model for perplexity: {eval_model_name} ...")
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
eval_model = AutoModelForCausalLM.from_pretrained(
    eval_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
eval_model.eval()
if device == "cuda":
    torch.cuda.empty_cache()
print("Eval model loaded.")

def perplexity_sliding_window(text, tokenizer, model, max_length=1024, stride=768):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    n_tokens = input_ids.size(0)
    if n_tokens <= 0:
        return float("nan")
    device_local = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for begin_loc in range(0, n_tokens, stride):
        end_loc = min(begin_loc + max_length, n_tokens)
        input_ids_window = input_ids[begin_loc:end_loc].unsqueeze(0).to(device_local)
        target_ids = input_ids_window.clone()
        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)
            nll_window = outputs.loss.item() * (end_loc - begin_loc)
        if begin_loc == 0:
            tokens_counted = end_loc - begin_loc
        else:
            tokens_counted = end_loc - begin_loc - (max_length - stride)
            if tokens_counted < 0:
                tokens_counted = 0
        total_nll += nll_window
        total_tokens += tokens_counted
        if end_loc == n_tokens:
            break

    if total_tokens <= 0:
        return float("nan")
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)

# ------------------------
# 5) evaluate mixtures
# ------------------------
human_ref_ngrams = ngram_dist(human_texts)
results = []

print("\n📊 Evaluating mixtures across α values and models ...")
for gm, synth_pool in synthetic_pools.items():
    print(f"\n=== Model: {gm} ===")
    for alpha in alphas:
        N = len(human_texts)
        num_synth = int(alpha * N)
        if num_synth <= len(synth_pool):
            synth_samples = random.sample(synth_pool, num_synth)
        else:
            synth_samples = [random.choice(synth_pool) for _ in range(num_synth)]

        mixed = random.sample(human_texts, N - num_synth) + synth_samples
        random.shuffle(mixed)
        pooled_text = " ".join(mixed)

        ppl = perplexity_sliding_window(pooled_text, eval_tokenizer, eval_model,
                                        max_length=max_length, stride=stride)
        kl = kl_divergence(human_ref_ngrams, ngram_dist(mixed))
        d2 = distinct_n(mixed)

        print(f"α={alpha:.2f} | Perplexity={ppl:.2f} | KL={kl:.5f} | Distinct-2={d2:.3f}")
        results.append((gm, alpha, float(ppl), float(kl), float(d2)))

# ------------------------
# 6) Save results + smooth d2
# ------------------------
df = pd.DataFrame(results, columns=["model","alpha","perplexity","kl_divergence","distinct2"])

# Smooth Distinct-2 per model across alpha
window = 3  # moving average window
df_smoothed = df.copy()
for model_name in df["model"].unique():
    mask = df["model"] == model_name
    subset = df.loc[mask].sort_values("alpha")
    smooth_d2 = uniform_filter1d(subset["distinct2"].values, size=window, mode="nearest")
    df_smoothed.loc[mask, "distinct2_smoothed"] = smooth_d2

df = df_smoothed
df.to_csv("ai_ouroboros_multi_model_ppl_fixed_smoothed.csv", index=False)
print("\n✅ Results saved to 'ai_ouroboros_multi_model_ppl_fixed_smoothed.csv'")

# ------------------------
# 7) Cleanup
# ------------------------
del eval_model
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
print("Done.")
