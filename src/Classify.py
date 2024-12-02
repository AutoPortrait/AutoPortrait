from LLM import LLMCurrent
from Prompts import Prompts
import warnings

prompts = Prompts()


def calculate_score(judgement: str) -> float:
    existence = [
        int(judgement.count("完全符合") > 0),
        int(judgement.count("大致符合") > 0),
        int(judgement.count("不太符合") > 0),
        int(judgement.count("完全不符合") > 0),
    ]
    count = sum(existence)
    if count == 0:
        warn = f'"{judgement}"不包含"完全符合"、"大致符合"、"不太符合"、"完全不符合"中的任何一个'
        warnings.warn(warn)
        return 0
    score = 0.0
    if existence[0] == 1:
        score += 3.0
    if existence[1] == 1:
        score += 1.0
    if existence[2] == 1:
        score -= 1.0
    if existence[3] == 1:
        score -= 3.0
    return score / count


def judge(portrait: tuple[str, str], interview: str) -> str:
    result = LLMCurrent.process(
        [
            {"role": "system", "content": prompts.prompt_classify},
            {"role": "user", "content": "以下是人物画像：" + portrait[1]},
            {"role": "user", "content": "以下是访谈记录：" + interview},
        ]
    )
    print(f"  - '{portrait[0]}'组: {result}")
    return result


def classify(original_portraits: list[tuple[str, str]], interview: str) -> list[int]:
    scores = [calculate_score(judge(portrait, interview)) for portrait in original_portraits]
    max_score = max(scores)
    ret = list[int]()
    for i, score in enumerate(scores):
        if score == max_score:
            ret.append(i)
    return ret
