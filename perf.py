import timeit
from functools import partial
from typing import Dict, List, Union

import torch
from torch.fx import GraphModule
from transformers import AutoModelForSequenceClassification
from transformers.utils.fx import symbolic_trace

from FxGraphOptimizer.passes import (DeadCodeEliminationPass,
                                     FuseDivIntoQKPass,
                                     MergeLinearWithSameSource,
                                     RemoveDropoutPass)

VOLCAB_SIZE = 30522

def prepare_bert_model() -> GraphModule:
    MODEL = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    graph = symbolic_trace(MODEL, input_names=["input_ids", "attention_mask", "token_type_ids"])
    return graph


def prepare_input_tensors(bs : int, seq_len : int, device : Union[torch.device, str] = 'cpu'):
    return {
        "input_ids" : torch.randint(VOLCAB_SIZE, (bs, seq_len), dtype=torch.long, device=device),
        "attention_mask" : torch.ones((bs, seq_len), dtype=torch.long, device=device),
        "token_type_ids" : torch.zeros((bs, seq_len), dtype=torch.long, device=device)
    }

def run_inference(model : GraphModule, inputs: Dict[str, torch.Tensor]):
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs


def benchmark(
    bs : int,
    seq_len : int,
    device : Union[torch.device, str] = 'cpu',
    repeat_times : int = 10
):
    inputs = prepare_input_tensors(bs, seq_len, device)
    fx_model = prepare_bert_model()
    
    fx_model.eval()
    fx_model.to(device)

    outputs_1 = run_inference(fx_model, inputs)

    ori_runtimes = timeit.repeat(
        partial(run_inference, model = fx_model, inputs = inputs),
        repeat=10,
        number=repeat_times,
    )

    for PASS in [
        FuseDivIntoQKPass,
        MergeLinearWithSameSource,
        RemoveDropoutPass,
        DeadCodeEliminationPass
    ]:
        fx_model = PASS()(fx_model)

    outputs_2 = run_inference(fx_model, inputs)
    opt_runtimes = timeit.repeat(
        partial(run_inference, model = fx_model, inputs = inputs),
        repeat=10,
        number=repeat_times,
    )

    assert torch.allclose(outputs_1["logits"], outputs_2["logits"], rtol = 1e-3, atol = 1e-6), "results mismatch!"

    print(f'bz={bs} seq_len={seq_len} device={device}')
    print(f"original runtime : {min(ori_runtimes) / repeat_times}s after optimization : {min(opt_runtimes) / repeat_times}s")


if __name__ == '__main__':
    for bz in [1, 4, 8, 16]:
      for seq_len in [8, 32, 64, 128, 256]:
        benchmark(bz, seq_len, device='cuda')