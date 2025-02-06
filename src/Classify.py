import asyncio
from LLM import LLMFast
from Prompts import Prompts
import warnings

multi_judge = 2

prompts = Prompts()


def calculate_score(judgement: str) -> float:
    existence = [
        judgement.count("完全符合"),
        judgement.count("大致符合"),
        judgement.count("不太符合"),
        judgement.count("完全不符合"),
    ]
    count = sum(existence)
    if count == 0:
        warn = f'"{judgement}"不包含"完全符合"、"大致符合"、"不太符合"、"完全不符合"中的任何一个'
        warnings.warn(warn)
        return 0
    score = 3.0 * existence[0] + 1.0 * existence[1] - 1.0 * existence[2] - 3.0 * existence[3]
    return score / count


async def judge_multi(portrait: tuple[str, str], interview: str) -> str:
    input = "[人物画像]\n" + portrait[1] + "\n\n[访谈记录]\n" + interview
    # results = LLMPrecise.batch_process(prompts.prompt_classify, input, multi_judge)
    results = [LLMFast.process(prompts.prompt_classify, input) for _ in range(multi_judge)]
    results = await asyncio.gather(*results)
    result = ", ".join(results)
    # print(f"  - '{portrait[0]}'组: {result}")
    return result


async def classify(original_portraits: list[tuple[str, str]], interview: str) -> list[int]:
    results = [judge_multi(portrait, interview) for portrait in original_portraits]
    results = await asyncio.gather(*results)
    scores = [calculate_score(result) for result in results]
    max_score = max(scores)
    ret = list[int]()
    for i, score in enumerate(scores):
        if score == max_score:
            ret.append(i)
    if len(ret) > 1:
        ret = []
    return ret
