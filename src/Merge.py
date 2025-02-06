from LLM import LLMPrecise
from Prompts import Prompts

prompts = Prompts()


def merge(original: str, additions: list[str]) -> str:
    input = (
        "[原文]\n\n"
        + original
        + "\n\n"
        + "\n\n".join([f"[补充信息 {i}]\n\n{addition}" for i, addition in enumerate(additions)])
    )
    return LLMPrecise.process(prompts.prompt_merge, input)
