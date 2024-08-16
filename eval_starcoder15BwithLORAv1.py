from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, filter_code, fix_indents
import os
import torch
from peft import PeftModel

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)
import torch

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> list[str]:
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
    out_path = "results/starcoder15BwithLORAv1/eval.jsonl"
    os.makedirs("results/starcoder15BwithLORAv1", exist_ok=True)


    checkpoint = "bigcode/starcoder"
    device = "cuda" # for GPU usage or "cpu" for CPU usage
    
    load_in_8bit = False
    
    # bitsandbytes config
    USE_NESTED_QUANT = True  # use_nested_quant
    BNB_4BIT_COMPUTE_DTYPE = "float16"  # bnb_4bit_compute_dtype
    
    # 4-bit quantization
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )
    
    # use_flash_attention_2:Support for Turing GPUs (T4, RTX 2080) is coming soon, please use FlashAttention 1.x for Turing GPUs for now.
    # from https://github.com/Dao-AILab/flash-attention
    base_model_4bit = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        # device_map=device_map,
        device_map="auto",
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    
    OUTPUT_DIR = "/content/LLM-Workshop2/personal_copilot/training/peft-lora-starcoder15B-v2-personal-copilot-H100-240GB-colab/checkpoint-200"  # output_dir
    # LORA ID
    peft_model_id = f"{OUTPUT_DIR}"
    
    base_model_4bit_with_lora_3 = PeftModel.from_pretrained(base_model_4bit, peft_model_id)
    base_model_4bit_with_lora_3.merge_and_unload()

    run_eval(
        base_model_4bit_with_lora_3,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
