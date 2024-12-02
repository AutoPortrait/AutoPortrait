from LLM import LLMCurrent, LLMProcessError
import os

path_interview_directory = "data/interview"
path_interview_censored_directory = "data/interview/censored"
path_interview_group_index = "data/interview/index-groups.txt"
path_initial_portrait = "prompt/初始人物画像.txt"
path_prompt_iterate = "prompt/迭代人物画像.txt"


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
                        censored_filename = f"{path_interview_censored_directory}/{filename}"
                        if os.path.exists(censored_filename) == False:
                            print(f"正在进行内容安全性预处理： {filename}")
                            with open(
                                f"{path_interview_directory}/{filename}", "r", encoding="utf-8"
                            ) as file:
                                data = file.read()
                                censored_data = censor(data)
                            with open(censored_filename, "w", encoding="utf-8") as file:
                                file.write(censored_data)
                        with open(censored_filename, "r", encoding="utf-8") as file:
                            self.list_data[len(self.list_data) - 1][1].append(file.read())
        with open(path_initial_portrait, "r", encoding="utf-8") as file:
            self.initial_portrait = file.read()
        with open(path_prompt_iterate, "r", encoding="utf-8") as file:
            self.prompt_iterate = file.read()
