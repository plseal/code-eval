from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval_one, filter_code, fix_indents
import os
import torch

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> list[str]:
    print("batch_size:",batch_size)
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.0,
        top_p=0.95,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    # fix_indents is required to fix the tab character that is generated from starcoder model
    return [filter_code(fix_indents(completion)) for completion in batch_completions]

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 1
    out_path = "results/phi3_3B_one/eval.jsonl"
    os.makedirs("results/phi3_3B_one", exist_ok=True)


    checkpoint = "microsoft/Phi-3-mini-128k-instruct"
    device = "cuda" # for GPU usage or "cpu" for CPU usage
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir = "/content").to(device)
    
    # multiple GPUs 
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir = "/content", trust_remote_code=True, device_map="auto")

    run_eval_one(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )