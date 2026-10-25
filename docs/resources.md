# Evaluation resources

EvoRouter reads model weights, datasets, and source dependencies from `assets/`. The download commands create this layout:

```text
assets/
├── data/
│   ├── budgetbench/
│   │   ├── train.inputs.jsonl
│   │   ├── train.labels.jsonl
│   │   ├── dev.inputs.jsonl
│   │   ├── dev.labels.jsonl
│   │   ├── test.inputs.jsonl
│   │   ├── test.labels.jsonl
│   │   └── assets/<task-id>/<image-file>
│   ├── data/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   └── test.parquet
│   └── ood/
│       ├── bbh_logical_deduction.jsonl
│       ├── livecodebench.jsonl
│       └── sources/livecodebench_test6.jsonl
├── weights/
│   ├── EvoRouter/
│   └── baselines/
└── upstream/
```

To use another location, add `--asset-root /path/to/assets` to the download, preparation, and evaluation commands, or set `EVOROUTER_ASSET_ROOT`.

## BudgetBench

Download [BudgetBench](https://huggingface.co/datasets/EvoRouter/BudgetBench) using a Hugging Face token with read access:

```bash
hf auth login
python -m evorouter download --data-repo EvoRouter/BudgetBench
```

The dataset includes five domains: question answering, mathematics, commonsense reasoning, knowledge, and multimodal understanding.

| Split | Tasks | JSONL file prefix |
|---|---:|---|
| Train | 10,500 | `train` |
| Validation | 1,000 | `dev` |
| Test | 1,300 | `test` |

The main evaluation reads `test.inputs.jsonl` and `test.labels.jsonl` under `assets/data/budgetbench/`. Each input record is paired with its label through the `id` field.

| File | Contents |
|---|---|
| `budgetbench/*.inputs.jsonl` | Task ID, query, task type, and Worker assets |
| `budgetbench/*.labels.jsonl` | Matching task ID, `answer`, `budgetbench_task`, and `budgetbench_condition` |
| `budgetbench/assets/<task-id>/<image-file>` | Images referenced by the task records |
| `data/*.parquet` | Tabular versions of the three splits |

Image paths are relative to the `budgetbench/` directory. Keep that directory's structure intact when copying the dataset. The paired JSONL files and image assets supply the evaluator inputs; the Parquet tables support dataset browsing and analysis.

## OOD data

Download BBH logical deduction from [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) and LiveCodeBench from its [official dataset repository](https://huggingface.co/datasets/livecodebench/code_generation_lite):

```bash
mkdir -p downloads
curl -L https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd481feebf959a6b59d61ea57bdcf30964d/bbh/logical_deduction_five_objects.json -o downloads/bbh_logical_deduction.json
hf download livecodebench/code_generation_lite test6.jsonl \
  --repo-type dataset \
  --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 \
  --local-dir downloads/livecodebench
python -m evorouter prepare-ood \
  --bbh-source downloads/bbh_logical_deduction.json \
  --livecodebench-source downloads/livecodebench/test6.jsonl
```

The preparation command uses the task selections in `configs/ood_items.json` and writes 200 BBH logical deduction tasks and 175 LiveCodeBench tasks to `assets/data/ood/`. It also places the LiveCodeBench source file under `assets/data/ood/sources/` for grading.

Install the grading source and run the OOD evaluation:

```bash
python -m evorouter download --sources
python -m evorouter evaluate \
  --methods evorouter \
  --suite ood \
  --output-dir outputs/evorouter-ood
```

To evaluate one OOD dataset, add `--datasets bbh_logical_deduction` or `--datasets livecodebench`.

## EvoRouter model

Download [EvoRouter-4B](https://huggingface.co/EvoRouter/EvoRouter-4B) with a Hugging Face token that has read access to the model repository:

```bash
hf auth login
python -m evorouter download --model-repo EvoRouter/EvoRouter-4B
```

The full model is stored under `assets/weights/EvoRouter/`. Keep the configuration, weights, tokenizer files, and chat template together:

```text
weights/EvoRouter/
├── config.json
├── generation_config.json
├── model-*.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── added_tokens.json
├── chat_template.jinja
├── merges.txt
└── vocab.json
```

To load a model directory stored elsewhere:

```bash
python -m evorouter evaluate \
  --methods evorouter \
  --model-path /path/to/EvoRouter \
  --output-dir outputs/evorouter
```

Place baseline weight files under `assets/weights/baselines/`, preserving their method directories and associated encoder files.

## Source dependencies

Download the routing implementations and LiveCodeBench grader:

```bash
python -m evorouter download --sources
```

The command installs the versions specified in `configs/upstream.json` under `assets/upstream/` and reuses existing source directories.

## Resource download options

The `download` command accepts Hugging Face repository IDs and places each resource directly into the matching local directory:

| Option | Repository type | Destination under the asset root |
|---|---|---|
| `--model-repo` | Model | `weights/EvoRouter/` |
| `--data-repo` | Dataset | `data/` |
| `--baseline-repo` | Model | `weights/baselines/` |
| `--upstream-repo` | Dataset | `upstream/` |

Use `--revision` to select a repository revision. Multiple resource options can be supplied in the same command.
