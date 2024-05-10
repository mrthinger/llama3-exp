# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from typing import List, Optional

import fire
from tqdm import tqdm

from llama import Dialog, Llama
from evalplus.data import get_human_eval_plus, write_jsonl, load_solutions

import re


def extract_md_blocks(text):
    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in matches]


def get_non_completed_problems(samples):
    non_completed_problems = [
        (task_id, problem)
        for task_id, problem in get_human_eval_plus().items()
        if not any(sample["task_id"] == task_id for sample in samples)
    ]
    return non_completed_problems


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"tokenizer_path: {tokenizer_path}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"max_batch_size: {max_batch_size}")
    print(f"max_gen_len: {max_gen_len}")
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # def GEN_SOLUTION(prompt: str):
    #     print(prompt)

    # samples = [
    #     dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"]))
    #     for task_id, problem in get_human_eval_plus().items()
    # ]
    # write_jsonl("samples.jsonl", samples)

    if not os.path.exists("samples.jsonl"):
        samples = []
    else:
        samples = list(load_solutions("samples.jsonl"))

    non_completed_problems = get_non_completed_problems(samples)

    for i in tqdm(range(0, len(non_completed_problems), max_batch_size)):
        batch_problems = non_completed_problems[i : i + max_batch_size]

        dialogs: List[Dialog] = [
            [{"role": "user", "content": problem["prompt"]}]
            for task_id, problem in batch_problems
        ]

        if not dialogs:
            continue

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for (task_id, problem), dialog, result in zip(batch_problems, dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
            result_code_blocks = extract_md_blocks(result["generation"]["content"])
            if len(result_code_blocks) == 0:
                print("\033[91mNO CODE FROM ASSISTANT!\033[0m")
            else:
                result_block = result_code_blocks[0]
                print("\033[92mCODE FROM ASSISTANT:\033[0m")
                print(result_block)

                samples.append(dict(task_id=task_id, solution=result_block))
                write_jsonl("samples.jsonl", samples)

            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
