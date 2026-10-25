# EvoRouter

**Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation**

EvoRouter brings cost-efficient multi-agent routing to a shared evaluation workflow. Run EvoRouter, five individual language models, and nine routing baselines on BudgetBench, BBH logical deduction, and LiveCodeBench. The evaluator records task predictions, accuracy, and Worker FLOPs, with consistent summaries across methods, datasets, and domains.

[Model weights](https://huggingface.co/EvoRouter/EvoRouter-4B) · [BudgetBench](https://huggingface.co/datasets/EvoRouter/BudgetBench) · [Resource guide](docs/resources.md)

## Installation

Use Python 3.11 and CUDA 12.8, then run the following commands:

```bash
git clone https://github.com/CenYangtze/EvoRouter.git
cd EvoRouter
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install the additional dependencies to run the routing baselines:

```bash
pip install -r requirements-baselines.txt
```

## Configure the model API

Create the API configuration:

```bash
cp .env.example .env
```

Set the endpoint and key in `.env`:

```dotenv
OPENAI_BASE_URL=https://your-api-host/v1
OPENAI_API_KEY=your-api-key
```

The OpenAI-compatible endpoint must serve these five Worker model IDs:

```text
qwen3.5-35b-a3b
qwen3.5-122b-a10b
deepseek-v4-flash
qwen3.5-397b-a17b
glm-5
```

You can also supply these settings through environment variables, which take precedence over `.env`.

## Download the model and dataset

Sign in with a Hugging Face token that has read access to both resource repositories, then download the files:

```bash
hf auth login
python -m evorouter download --model-repo EvoRouter/EvoRouter-4B
python -m evorouter download --data-repo EvoRouter/BudgetBench
```

The model is placed under `assets/weights/EvoRouter/` and BudgetBench under `assets/data/budgetbench/`. BudgetBench contains 10,500 training tasks, 1,000 validation tasks, and 1,300 test tasks across five domains. The main evaluation uses the test split.

Download the source dependencies used by the routing baselines and the LiveCodeBench grader:

```bash
python -m evorouter download --sources
```

Prepare BBH logical deduction and LiveCodeBench with the [official download commands](docs/resources.md#ood-data). The OOD evaluation contains 200 BBH tasks and 175 LiveCodeBench tasks.

Use `--asset-root /path/to/assets` with download, preparation, and evaluation commands to choose another resource directory. `EVOROUTER_ASSET_ROOT` sets the default location. The [resource guide](docs/resources.md) describes the files and directory layout.

## Training

Install the training dependencies:

```bash
pip install -r requirements-training.txt
```

Use a Coordinator SFT checkpoint and its `sft_manifest.json`. Start RL with the BudgetBench training data:

```bash
torchrun --standalone --nproc_per_node=1 -m evorouter.training rl \
  --model outputs/sft/best \
  --sft-source outputs/sft/best/sft_manifest.json \
  --teacher-model Qwen/Qwen3-32B-AWQ \
  --data-dir assets/data/budgetbench \
  --output-dir outputs/training
```

The configured API endpoint must also serve `deepseek-v4-pro-thinking` for Insight updates. Replace the model and data paths with your local paths as needed. Add `--resume` to the same command to continue from the latest checkpoint. The completed model is saved to `outputs/training/final` and can be passed to evaluation with `--model-path`.

SFT data collection and SFT training have separate entrypoints:

```bash
python -m evorouter.training collect --help
python -m evorouter.training sft --help
```

## Run EvoRouter

Evaluate the complete BudgetBench test split:

```bash
python -m evorouter evaluate \
  --methods evorouter \
  --output-dir outputs/evorouter
```

Evaluate BBH logical deduction and LiveCodeBench:

```bash
python -m evorouter evaluate \
  --methods evorouter \
  --suite ood \
  --output-dir outputs/evorouter-ood
```

Use `--suite all` to run BudgetBench and both OOD datasets. To load EvoRouter from another local directory, add `--model-path /path/to/EvoRouter`.

## Run baselines

Place the baseline weight files under `assets/weights/baselines/`, preserving their method directories. Run several methods in one command:

```bash
python -m evorouter evaluate \
  --methods vanilla routellm graphrouter routerdc router-r1 \
  --output-dir outputs/baselines
```

`vanilla` runs all five individual Worker models. Select a single model with its full ID, such as `vanilla__qwen3.5-35b-a3b`.

| Routing baseline | Method argument |
|---|---|
| RouteLLM | `routellm` |
| GraphRouter | `graphrouter` |
| RouterDC | `routerdc` |
| Router-R1 | `router-r1` |
| xRouter | `xrouter` |
| DAAO | `daao` |
| MasRouter | `masrouter` |
| RCR-Router | `rcr-router` |
| AgentRouter | `agentrouter` |

Fixed configurations are available as `cot`, `reflection`, `chain`, `debate`, `fullconnected`, `tree`, and `sc`.

Run EvoRouter, the five individual models, and all nine routing baselines across both suites:

```bash
python -m evorouter evaluate \
  --methods all \
  --suite all \
  --output-dir outputs/all
```

Methods run sequentially in separate processes, releasing local model memory between runs.

## Select tasks and configure a run

Select datasets and set task concurrency and EvoRouter generation batch size:

```bash
python -m evorouter evaluate \
  --methods evorouter \
  --datasets math mmlu_pro \
  --concurrency 4 \
  --batch-size 2 \
  --output-dir outputs/selected
```

| Option | Default | Description |
|---|---|---|
| `--model-path` | `(asset root)/weights/EvoRouter` | Local EvoRouter model directory |
| `--suite` | `main` | `main`, `ood`, or `all` |
| `--datasets` | All datasets in the selected suite | Space-separated dataset names |
| `--concurrency` | `16` | Concurrent evaluation tasks |
| `--batch-size` | `8` | EvoRouter generation batch size |
| `--device` | Automatic | Device for the local EvoRouter model |
| `--limit` | `0` | First N selected tasks per suite; `0` runs all |
| `--per-domain` | `0` | Sample N tasks from each domain; `0` uses the full selection |
| `--seed` | `20260906` | Task sampling and EvoRouter seed |

Repeat the same command to resume a run. Completed task IDs are skipped. Choose a new output directory when changing the model, dataset selection, or evaluation settings.

## Read the results

Each method writes its settings, task records, and aggregate metrics. The top-level summaries combine all selected methods:

```text
outputs/all/
├── evorouter/
│   ├── config.json
│   ├── results.jsonl
│   └── summary.json
├── routellm/
│   ├── config.json
│   ├── results.jsonl
│   └── summary.json
├── summary.json
└── summary.csv
```

`results.jsonl` contains predictions, scores, model calls, and execution costs. The summaries report accuracy and mean Worker FLOPs (`mean_worker_flops`) for each suite, dataset, and domain.

Regenerate the JSON and CSV summaries from existing task records:

```bash
python -m evorouter summarize --output-dir outputs/all
```

## License

[Apache 2.0](LICENSE). Third-party notices are listed in [NOTICE](NOTICE).
