from LLM import LLMCurrent, LLMProcessError
import os

path_interview_directory = "data/interview"
path_interview_censored_directory = "data/interview/censored"
path_interview_group_index = "data/interview/index-groups.txt"
path_interview_uncertain_index = "data/interview/index-uncertain.txt"


def censor(text: str) -> str:
    if len(text) == 0:
        return text
    try:
        print(f"检查 {len(text)} 字 ... ", end="", flush=True)
        LLMCurrent.process(text)
        print("合规")
        return text
    except LLMProcessError:
        print("违规")
        lines = text.splitlines()
        if len(lines) == 1:
            return f"[根据有关法律法规省略 {len(lines[0])} 字]\n"
        if len(lines) > 1:
            middle = len(lines) // 2
            censored_data = ""
            censored_data += censor("\n".join(lines[:middle]))
            censored_data += "\n"
            censored_data += censor("\n".join(lines[middle:]))
            return censored_data
        raise BaseException("Impossible")


def load_interview(filename: str) -> str:
    censored_filename = f"{path_interview_censored_directory}/{filename}"
    if os.path.exists(censored_filename) == False:
        print(f"正在进行内容安全性预处理： {filename}")
        with open(f"{path_interview_directory}/{filename}", "r", encoding="utf-8") as file:
            data = file.read()
            censored_data = censor(data)
        with open(censored_filename, "w", encoding="utf-8") as file:
            file.write(censored_data)
    with open(censored_filename, "r", encoding="utf-8") as file:
        return file.read()


class Input:
    def __init__(self):
        self.list_data = list[str, list[str]]()  # group name, data list
        with open(path_interview_group_index, "r", encoding="utf-8") as group_index_file:
            group_index_lines = group_index_file.read().splitlines()
            group_index_pairs = [line.split(sep=",") for line in group_index_lines]
            for group_index_pair in group_index_pairs:
                self.list_data.append([group_index_pair[0], []])
                with open(
                    f"{path_interview_directory}/{group_index_pair[1]}", "r", encoding="utf-8"
                ) as group_index_file:
                    filenames = group_index_file.read().splitlines()
                    for filename in filenames:
                        self.list_data[len(self.list_data) - 1][1].append(load_interview(filename))
        self.uncertain_data = []
        with open(path_interview_uncertain_index, "r", encoding="utf-8") as uncertain_index_file:
            filenames = uncertain_index_file.read().splitlines()
            for filename in filenames:
                self.uncertain_data.append(load_interview(filename))
