from LLM import LLMCurrent
from Prompts import Prompts
import warnings


def remove_metadata(text: str) -> str:
    # Remove "文字记录:" and anything before it
    return text[text.find("文字记录:") + 5 :].strip()


def to_list(text: str) -> list[str]:
    # Split by empty line
    result = text.split("\n\n")
    return [entry.strip() for entry in result]


class InterviewEntry:
    def __init__(self, entry_str: str):
        lines = entry_str.split("\n")
        meta = lines[0].strip().split(" ")
        self.person = meta[0]
        self.time = meta[1]
        self.text = lines[1].strip()

    def __str__(self):
        return f"{self.person} {self.time}\n{self.text}\n"


prompts = Prompts()


def split(original_text: str) -> list[str]:
    entry_str_list = to_list(remove_metadata(original_text))
    entry_list = [InterviewEntry(entry_str) for entry_str in entry_str_list]
    while True:
        times_str = LLMCurrent.process(prompts.prompt_cut_interview, original_text)
        times_list = times_str.strip().split("\n")
        fail = False
        for time in times_list:
            match = list(filter(lambda x: x.time == time, entry_list))
            if len(match) == 0:
                warnings.warn(f"LLM returns time '{time}' which is not found. Retry.")
                fail = True
                break
            if len(match) > 1:
                warnings.warn(f"Multiple entries found for time '{time}'. No retry.")
        if entry_list[0].time != times_list[0]:
            warnings.warn(f"First entry time '{times_list[0]}' does not match. Retry.")
            fail = True
        if not fail:
            result = []
            last = None
            for entry in entry_list:
                if entry.time in times_list:
                    result.append(str(entry))
                    if last:
                        result[-1] = last + "\n" + result[-1]
                else:
                    result[-1] += "\n" + str(entry)
                last = str(entry)
            return result


def main():
    from Input import Input

    input = Input()
    example = input.uncertain[0][1]
    result = split(example)
    for entry in enumerate(result):
        print(f"[Entry {entry[0]}]")
        print(entry[1])


if __name__ == "__main__":
    main()
