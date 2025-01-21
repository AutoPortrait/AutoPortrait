from LLM import LLMCurrent
from Prompts import Prompts
from tqdm import tqdm

prompts = Prompts()


def merge_interview_segments(interview_segments: list[str], progress=True) -> str:
    results = ""
    for i in tqdm(
        range(len(interview_segments)),
        desc="Analyzing segments",
        leave=False,
        disable=not progress,
    ):
        result = LLMCurrent.process(
            prompts.prompt_analyze_interview_segments, interview_segments[i]
        )
        results += f"[特点描述 {i+1}]\n{result}\n\n"
    return results


def main():
    from Input import Input
    from Split import split

    input = Input()
    example = input.interviews[0].data
    result = merge_interview_segments(split(example))
    print("Result:")
    print(result)


if __name__ == "__main__":
    main()
