from __future__ import annotations
import importlib.util, sys
import torch
from .prepare import ROOT
from .core import task_text


def original_model():
    sys.path.insert(0, str(ROOT / "deps"))
    path = ROOT / "upstream/GraphRouter/model/graph_nn.py"
    spec = importlib.util.spec_from_file_location("paper_original_graph_nn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EncoderDecoderNet


def embeddings(items, encoder):
    q = encoder.encode(
        [task_text(i) for i in items], show_progress_bar=False, convert_to_numpy=True
    )
    t = encoder.encode(
        [i["dataset"] for i in items], show_progress_bar=False, convert_to_numpy=True
    )
    return (torch.tensor(q), torch.tensor(t))
