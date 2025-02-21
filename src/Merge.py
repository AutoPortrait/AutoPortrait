from LLM import LLMInstructional
from Prompts import Prompts

prompts = Prompts()


async def merge(original: str, additions: list[str]) -> str:
    input = (
        "[原文]\n\n"
        + original
        + "\n\n"
        + "\n\n".join([f"[补充信息 {i}]\n\n{addition}" for i, addition in enumerate(additions)])
    )
    return await LLMInstructional.process(prompts.prompt_merge, input)
