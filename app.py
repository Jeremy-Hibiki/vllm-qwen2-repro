from enum import Enum
import logging
from typing import cast
import typer
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, Qwen2Tokenizer

MODEL = "/data/models/Qwen2-7B-Instruct-AWQ"


class LogLevel(str, Enum):
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


def main(use_fp8_cache: bool = False, loglevel: LogLevel = LogLevel.INFO):
    logging.getLogger("vllm").setLevel(logging._nameToLevel.get(loglevel, 20))

    tokenizer = cast(Qwen2Tokenizer, AutoTokenizer.from_pretrained(MODEL))
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)
    llm = LLM(
        model=MODEL,
        kv_cache_dtype="fp8" if use_fp8_cache else "auto",
        gpu_memory_utilization=0.6,
        swap_space=0,
        max_model_len=4096,
        max_seq_len_to_capture=4096,
    )
    inputs = str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Tell me about the history of Machine Learning."}],
            tokenize=False,
        )
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for output in outputs:
        print("=" * 80)
        print(output.outputs[0].text)
        print("=" * 80)
        print(f"TOKENS: {len(output.outputs[0].token_ids)}")


if __name__ == "__main__":
    typer.run(main)
