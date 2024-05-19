import torch
from safetensors import safe_open
import os
from fairscale.nn.model_parallel.utils import VocabUtility
import glob

from tqdm import tqdm


def get_model_parallel_world_size():
    # Replace this with the actual function to get the model parallel world size
    return 8


def divide_and_check_no_remainder(numerator, denominator):
    assert (
        numerator % denominator == 0
    ), f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


def convert_safetensors_to_pth(safetensors_dir, pth_output_dir):
    model_parallel_world_size = get_model_parallel_world_size()

    os.makedirs(pth_output_dir, exist_ok=True)

    safetensors_paths = glob.glob(os.path.join(safetensors_dir, "*.safetensors"))

    tensors = {}
    for safetensors_path in safetensors_paths:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    for model_parallel_rank in tqdm(range(model_parallel_world_size)):
        shard_tensors = {}
        for key, tensor in tensors.items():
            if (
                "tok_embeddings.weight" in key
                or "output.weight" in key
                or "embed_tokens.weight" in key
            ):
                # Shard along the vocabulary dimension
                vocab_start_index, vocab_end_index = (
                    VocabUtility.vocab_range_from_global_vocab_size(
                        tensor.size(0), model_parallel_rank, model_parallel_world_size
                    )
                )
                tensor_shard = tensor[vocab_start_index:vocab_end_index].clone()
            elif (
                "wq.weight" in key
                or "wk.weight" in key
                or "wv.weight" in key
                or "w1.weight" in key
                or "w3.weight" in key
                or "q_proj.weight" in key
                or "k_proj.weight" in key
                or "v_proj.weight" in key
                or "gate_proj.weight" in key
                or "down_proj.weight" in key
                or "up_proj.weight" in key
            ):
                # Column parallelism
                partition_dim = 0
                output_size_per_partition = divide_and_check_no_remainder(
                    tensor.size(partition_dim), model_parallel_world_size
                )
                start_index = model_parallel_rank * output_size_per_partition
                end_index = start_index + output_size_per_partition
                tensor_shard = tensor[start_index:end_index, :].clone()
            elif "wo.weight" in key or "w2.weight" in key or "o_proj.weight" in key:
                # Row parallelism
                partition_dim = 1
                input_size_per_partition = divide_and_check_no_remainder(
                    tensor.size(partition_dim), model_parallel_world_size
                )
                start_index = model_parallel_rank * input_size_per_partition
                end_index = start_index + input_size_per_partition
                tensor_shard = tensor[:, start_index:end_index].clone()
            else:
                tensor_shard = tensor

            shard_tensors[key] = tensor_shard

        pth_output_path = os.path.join(
            pth_output_dir, f"model-{model_parallel_rank}.pth"
        )
        torch.save(shard_tensors, pth_output_path)
        print(f"Converted shard {model_parallel_rank} to {pth_output_path}")


# Example usage
safetensors_dir = "Smaug-Llama-3-70B-Instruct"
pth_output_dir = "70b"
convert_safetensors_to_pth(safetensors_dir, pth_output_dir)
