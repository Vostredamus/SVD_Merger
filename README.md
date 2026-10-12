# LLM Traits

**Measure how fine-tuned models differ from their common base, then merge them on purpose.**

LLM Traits decomposes a family of fine-tuned LLMs into their differences from a shared base model, finds the main directions those differences share, and uses that map to build a new model: keep one model's identity, and graft in only the parts of other models that are genuinely new and do not conflict with it.

Two models built with this tool:

| Model | Departure (identity kept) | Donors |
|---|---|---|
| [**ArcOpening/Animaculus**](https://huggingface.co/ArcOpening/Animaculus) | Darkhn/L3.3-70B-Animus-V12.5 | 12 community fine-tunes |
| [**ArcOpening/Vulspeculatius**](https://huggingface.co/ArcOpening/Vulspeculatius) | Sao10K/Llama-3.3-70B-Vulpecula-r1 | 6 donors chosen for complementarity |

How each was designed is described in [EXAMPLES.md](EXAMPLES.md).

---

## The concept

### 1. Everything is measured against one base model

All models analyzed here are fine-tunes of Llama 3.3 70B Instruct. The common reference (**BASE**) used for both models above is:

> **[kldzj/Llama-3.3-70B-Instruct-heretic](https://huggingface.co/kldzj/Llama-3.3-70B-Instruct-heretic)** — an abliterated variant of `meta-llama/Llama-3.3-70B-Instruct`

Every model becomes a **delta**:

```
delta_i = model_i − BASE
```

A delta is "what the fine-tune changed". Because all deltas start from the same origin, they can be compared and combined with ordinary linear algebra. (Models fine-tuned from a different base, e.g. Llama 3.1, can still be analyzed, but part of their delta then reflects the base difference rather than the fine-tune.)

### 2. SVD finds the shared directions

Stack all deltas as rows of one matrix `D` and take its **singular value decomposition (SVD)**:

```
D = U · S · Vᵀ
```

In plain terms, SVD rewrites a set of vectors as combinations of a few orthogonal **directions** (`V`), ordered by how much of the total change each one explains (`S`). Every model gets a **coefficient** on each direction (`U·S`): its sign says which side of the direction the model sits on, its size says how strongly.

For 70B models the deltas have ~70 billion entries, so the SVD is computed through the small model-by-model Gram matrix `D·Dᵀ`, streamed tensor by tensor. The same Gram matrices are also computed per block of weights (an "atlas"), which shows *where* in the network each model's changes live.

### 3. Why this helps merging

Common merge methods treat the problem globally:

- **Linear averaging / SLERP** blend whole models with one weight; everything is diluted together.
- **Task arithmetic, TIES, DARE** work on deltas too, but decide per parameter with fixed rules (trim small values, vote on signs, drop at random) without looking at how the models relate to each other.

LLM Traits adds measurement before merging:

- **A shared coordinate system.** You see which models are redundant, which are genuinely different, and which are outliers *before* spending hours on a merge.
- **Protect the departure model.** The blocks carrying most of the departure model's own fine-tuning energy are left untouched, so its identity survives.
- **Novelty instead of averaging.** Donors only add what the departure model and earlier donors do not already contain (each donor's delta is projected away from what is already there).
- **Local conflict detection.** Blocks where a donor points against the departure model are rejected, block by block.
- **Routing by region.** Donors compete per block; each region of the network is written by the donor that is strongest there, optionally restricted to chosen layers and projection types.
- **Bounded updates.** The size of every change is capped relative to the departure model's weights.

Geometry is a measurement, not a guarantee of behavior: a direction can be novel and safe to add and still not change outputs the way you hope. The analysis tells you where to look; testing the model tells you what you got.

---

## Installation

```bash
git clone <this repository>
cd LLM_traits
pip install -r requirements.txt
```

Requires Python 3.10+, `torch`, `safetensors`, `numpy`, `matplotlib`, `tqdm`, and `streamlit` (GUI). Analysis and composition run on CPU. Optional quantization to EXL3 needs a separate [ExLlamaV3](https://github.com/turboderp-org/exllamav3) install and a GPU.

Disk space matters: each 70B delta is ~130 GB in float16, and a composed model is ~140 GB.

Optional environment variables (all paths can also be given on the command line or in the GUI):

| Variable | Meaning |
|---|---|
| `LLMTRAITS_BASE` | path to the BASE checkpoint |
| `LLMTRAITS_DELTAS_DIR` | where deltas are written (default `./deltas`) |
| `LLMTRAITS_CONVERTER_PYTHON` | Python executable of your ExLlamaV3 environment |
| `LLMTRAITS_CONVERT_SCRIPT` | path to ExLlamaV3's `convert.py` |

---

## Usage

### Analysis (command line)

```bash
# 1. Delta of each fine-tune against BASE
python -m llmtraits delta <model_dir> --base <base_dir> --output-dir deltas/<name>

# 2. Cross-model analysis and global SVD (plots included)
python -m llmtraits cross deltas/ModelA deltas/ModelB ... --output-dir analysis

# 3. Block-level atlas (once), then metrics for a chosen departure model
python -m llmtraits atlas-scan    --analysis analysis --output-dir atlas
python -m llmtraits atlas-analyze --atlas-dir atlas --analysis analysis \
                                  --departure-label <model_name>
```

Step 3 writes donor scores, territory maps, and automatically generated candidate recipes. Re-running `atlas-analyze` with another `--departure-label` is fast, since it reuses the cached atlas, so several departure candidates can be compared cheaply.

Long jobs (`svd-launch`, `atlas-launch`, `atlas-analyze-launch`) run as detached processes and every stage is resumable after an interruption.

### Building a model (GUI)

```bash
python -m llmtraits gui --analysis analysis
```

The GUI opens in the browser with the departure model selectable in the sidebar and five tabs:

| Tab | What it does |
|---|---|
| **Donor transfer** | Move the departure model's coefficient on one SVD direction toward a donor's (`gamma` = 0 keeps it, 1 matches the donor) |
| **Direct steering** | Set a coefficient explicitly (add, set, or multiply) |
| **Overlap & islands** | How each direction overlaps the departure model; where donors have novel signal |
| **Routed composition** | Build a protected, block-routed recipe: donors, priorities, layer/projection bands, protection level |
| **Build & compress** | Write the recipe, compose the full model, and optionally quantize it with ExLlamaV3 as one resumable background job |

A recipe can also be composed from the command line:

```bash
python -m llmtraits route-preview --recipe recipe.json   # check routing first
python -m llmtraits compose-svd   --recipe recipe.json
```

### Main outputs

| File | Content |
|---|---|
| `model_cosine_similarity.png` | how similar each pair of deltas is |
| `svd_explained_energy.png` | how much each SVD direction explains |
| `svd_model_coefficients.png` | each model's position on each direction |
| `analysis_summary.json` | per-donor novelty, conflict, size and score for the chosen departure |
| `donor_territories.csv`, `plots/donor_territory.png` | which donor is strongest in each layer and projection type |
| `candidate_report.md` | automatically proposed recipes |

What these look like and how they were used to design the two models above: [EXAMPLES.md](EXAMPLES.md).

---

## Tests

The tests are not needed to use the package (nothing in `llmtraits/` imports them). They check the math and the bookkeeping on small synthetic models — exact reconstruction of a model from BASE + delta, SVD reconstruction, routing and protection rules, resuming after interruption, and refusal to reuse stale results. No downloads or GPU needed; the suite runs in about a minute.

```bash
python -m pytest tests
```

The delta writer refuses to start with less than 10 GB free, even for tiny test models. If your temp drive is short on space, point pytest elsewhere: `python -m pytest tests --basetemp=<dir on a drive with space>`.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). Copyright 2026 ArcOpening.

The license covers the code only. Models built with this tool are derivatives of their base and donor models: they are governed by the [Llama 3.3 Community License](https://www.llama.com/llama3_3/license/) and by the licenses of every donor model used.

## Limitations

- Weight-space geometry does not measure writing quality, safety, or style; every merge still needs to be tested by generating text.
- All models must share the same architecture and tensor shapes as BASE.
- Large disk and time cost for 70B models (hours per analysis or composition step).
