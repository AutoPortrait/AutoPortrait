from LLM import LLMCurrent, LLMProcessError
import os

path_input_directory = "input"
path_censored_interview_directory = "input/censored"
path_groups_index = "input/groups.txt"
path_interviews_index = "input/interviews.txt"


class Interview:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = Interview.load_interview(filename)

    def censor(text: str) -> str:
        if len(text) == 0:
            return text
        try:
            print(f"检查 {len(text)} 字 ... ", end="", flush=True)
            LLMCurrent.process("", text)
            print("合规")
            return text
        except LLMProcessError as e:
            if str(e).find("不安全或敏感内容") == -1:
                raise e
            print("违规")
            lines = text.splitlines()
            if len(lines) == 1:
                return f"[根据有关法律法规省略 {len(lines[0])} 字]\n"
            if len(lines) > 1:
                middle = len(lines) // 2
                censored_data = ""
                censored_data += Interview.censor("\n".join(lines[:middle]))
                censored_data += "\n"
                censored_data += Interview.censor("\n".join(lines[middle:]))
                return censored_data
            raise BaseException("Impossible")

    def load_interview(filename: str) -> str:
        censored_filename = f"{path_censored_interview_directory}/{filename}"
        if os.path.exists(censored_filename) == False:
            print(f"正在进行内容安全性预处理： {filename}")
            with open(f"{path_input_directory}/{filename}", "r", encoding="utf-8") as file:
                data = file.read()
                censored_data = Interview.censor(data)
            with open(censored_filename, "w", encoding="utf-8") as file:
                file.write(censored_data)
        with open(censored_filename, "r", encoding="utf-8") as file:
            return file.read()


class Group:
    def __init__(
        self, name: str, initial_portrait_filename: str, initial_interviews_index_filename: str
    ):
        self.name = name
        initial_portrait_path = f"{path_input_directory}/{initial_portrait_filename}"
        with open(initial_portrait_path, "r", encoding="utf-8") as portrait_file:
            self.portrait = portrait_file.read()
        self.initial_interviews = list[Interview]()
        initial_interviews_index_path = (
            f"{path_input_directory}/{initial_interviews_index_filename}"
        )
        with open(initial_interviews_index_path, "r", encoding="utf-8") as index_file:
            filenames = index_file.read().splitlines()
            for filename in filenames:
                self.initial_interviews.append(Interview(filename))


class Input:
    def __init__(self):
        self.groups = list[Group]()
        with open(path_groups_index, "r", encoding="utf-8") as groups_index_file:
            group_index_lines = groups_index_file.read().splitlines()
            for group_index_line in group_index_lines:
                group_index_line_split = group_index_line.split(",")
                self.groups.append(
                    Group(
                        name=group_index_line_split[0],
                        initial_portrait_filename=group_index_line_split[1],
                        initial_interviews_index_filename=group_index_line_split[2],
                    )
                )
        self.interviews = list[Interview]()
        with open(path_interviews_index, "r", encoding="utf-8") as interviews_index_file:
            filenames = interviews_index_file.read().splitlines()
            for filename in filenames:
                self.interviews.append(Interview(filename))


if __name__ == "__main__":
    input = Input()
    print(input.groups)
    print(input.interviews)
