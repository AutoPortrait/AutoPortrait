from LLM import LLMCurrent
from Prompts import Prompts

prompts = Prompts()


def merge_interview_segments(interview_segments: list[str]) -> str:
    with_header = [f"[访谈片段 {i+1}]\n\n{segment}" for i, segment in enumerate(interview_segments)]
    input = "\n\n".join(with_header)
    return LLMCurrent.process(prompts.prompt_merge_interview_segments, input)


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
