path_initial_portrait = "prompt/初始人物画像.txt"
path_prompt_iterate = "prompt/迭代人物画像.txt"


class Prompts:
    def __init__(self):
        with open(path_initial_portrait, "r", encoding="utf-8") as file:
            self.initial_portrait = file.read()
        with open(path_prompt_iterate, "r", encoding="utf-8") as file:
            self.prompt_iterate = file.read()
