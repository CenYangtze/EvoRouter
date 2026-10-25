"""Download published resources into the evaluation directory layout."""

from pathlib import Path
import json
import subprocess
import tempfile
from urllib.request import Request, urlopen


def download(
    asset_root,
    data_repo=None,
    model_repo=None,
    baseline_repo=None,
    upstream_repo=None,
    revision=None,
):
    from huggingface_hub import snapshot_download

    root = Path(asset_root).expanduser().resolve()
    selections = (
        (data_repo, "dataset", "data"),
        (model_repo, "model", "weights/EvoRouter"),
        (baseline_repo, "model", "weights/baselines"),
        (upstream_repo, "dataset", "upstream"),
    )
    for repository, kind, folder in selections:
        if repository:
            snapshot_download(
                repo_id=repository,
                repo_type=kind,
                revision=revision,
                local_dir=root / folder,
            )
            print(f"{repository} -> {root / folder}")


def download_sources(asset_root):
    root = Path(asset_root).expanduser().resolve() / "upstream"
    sources = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/upstream.json").read_text()
    )
    root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = root / name
        if "files" in source:
            for relative, url in source["files"].items():
                target = destination / relative
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=60) as response:
                    content = response.read()
                target.write_bytes(content)
        elif not destination.exists():
            with tempfile.TemporaryDirectory(
                prefix=".download-", dir=root
            ) as temporary:
                checkout = Path(temporary) / name
                subprocess.run(["git", "init", str(checkout)], check=True)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "remote",
                        "add",
                        "origin",
                        source["url"],
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "fetch",
                        "--depth",
                        "1",
                        "origin",
                        source["revision"],
                    ],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
                    check=True,
                )
                checkout.rename(destination)
        print(f"{name} -> {destination}")
