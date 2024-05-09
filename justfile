gen:
    PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun --nproc_per_node 1 example_chat_completion.py \
        --max_seq_len 4096 \
        --max_batch_size 6 \
        --temperature 0 \
        --ckpt_dir Meta-Llama-3-8B-Instruct/ \
        --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model         

eval:
    docker run -v $(pwd):/app ganler/evalplus:latest --dataset humaneval --samples samples.jsonl