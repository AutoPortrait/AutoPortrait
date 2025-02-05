from LLM import LLMPrecise
from Prompts import Prompts
import warnings
import re


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
    tried_cnt = 0
    while True:
        tried_cnt += 1
        entry_min = len(entry_list) / 10
        entry_max = len(entry_list) / 3
        prompt = prompts.prompt_cut_interview + "\n(请切割 %d 到 %d 个片段）" % (
            entry_min,
            entry_max,
        )
        times_str = LLMPrecise.process(prompt, original_text)
        times_list = [s.strip() for s in times_str.strip().split("\n")]
        times_list = list(filter(lambda x: re.match(r"^\d{2}:\d{2}$", x), times_list))
        if (len(times_list) + 1 < entry_min or len(times_list) + 1 > entry_max) and tried_cnt < 5:
            warnings.warn(
                f"Number of entries {len(times_list)} out of range {entry_min}-{entry_max}. Retry."
            )
            continue
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
                if entry.time in times_list and (last is None or entry.time != last.time):
                    result.append(str(entry))
                    if last:
                        result[-1] = str(last) + "\n" + result[-1]
                else:
                    result[-1] += "\n" + str(entry)
                last = entry
            return result


def main():
    from Input import Input

    input = Input()
    example = input.interviews[0].data
    result = split(example)
    for entry in enumerate(result):
        print(f"[Entry {entry[0]}]")
        print(entry[1])


if __name__ == "__main__":
    main()
