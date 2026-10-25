"""Command-line entry point for downloading resources and running evaluations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(prog="evorouter")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("evaluate", help="Evaluate one or more methods")
    evaluate.add_argument("--methods", nargs="+", default=["evorouter"])
    evaluate.add_argument("--suite", choices=["main", "ood", "all"], default="main")
    evaluate.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.getenv("EVOROUTER_ASSET_ROOT", REPO_ROOT / "assets")),
    )
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--datasets", nargs="+")
    evaluate.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate the first N selected items; 0 uses the full split",
    )
    evaluate.add_argument(
        "--per-domain",
        type=int,
        default=0,
        help="Sample N items per domain with the evaluation seed",
    )
    evaluate.add_argument("--concurrency", type=int, default=16)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--model-path", type=Path, default=None)
    evaluate.add_argument("--seed", type=int, default=20260906)
    evaluate.add_argument("--worker-method", help=argparse.SUPPRESS)
    evaluate.add_argument("--worker-suite", help=argparse.SUPPRESS)

    download = commands.add_parser("download", help="Download Hugging Face resources")
    download.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.getenv("EVOROUTER_ASSET_ROOT", REPO_ROOT / "assets")),
    )
    download.add_argument("--data-repo")
    download.add_argument("--model-repo")
    download.add_argument("--baseline-repo")
    download.add_argument("--upstream-repo")
    download.add_argument("--revision")
    download.add_argument(
        "--sources",
        action="store_true",
        help="Download baseline and code-grader source dependencies",
    )

    prepare = commands.add_parser(
        "prepare-ood", help="Prepare official BBH and LiveCodeBench files"
    )
    prepare.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.getenv("EVOROUTER_ASSET_ROOT", REPO_ROOT / "assets")),
    )
    prepare.add_argument("--bbh-source", type=Path)
    prepare.add_argument("--livecodebench-source", type=Path)

    report = commands.add_parser("summarize", help="Create CSV and JSON summaries")
    report.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))

    args = parser.parse_args()
    if args.command == "evaluate":
        if (
            args.concurrency < 1
            or args.batch_size < 1
            or args.limit < 0
            or args.per_domain < 0
        ):
            parser.error(
                "Concurrency and batch size must be positive; sample counts must be nonnegative"
            )
        from .runner import evaluate

        evaluate(args)
    elif args.command == "download":
        if not any(
            (
                args.data_repo,
                args.model_repo,
                args.baseline_repo,
                args.upstream_repo,
                args.sources,
            )
        ):
            parser.error("Select a resource repository or --sources")
        from .artifacts import download, download_sources

        download(
            args.asset_root,
            data_repo=args.data_repo,
            model_repo=args.model_repo,
            baseline_repo=args.baseline_repo,
            upstream_repo=args.upstream_repo,
            revision=args.revision,
        )
        if args.sources:
            download_sources(args.asset_root)
    elif args.command == "prepare-ood":
        if not args.bbh_source and not args.livecodebench_source:
            parser.error("Select --bbh-source or --livecodebench-source")
        from .prepare_ood import prepare

        prepare(args.asset_root, args.bbh_source, args.livecodebench_source)
    else:
        from .results import summarize

        summarize(args.output_dir)
