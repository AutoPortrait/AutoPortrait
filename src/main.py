from datetime import datetime
import os
from LLM import LLMCurrent
from Input import Input
from Prompts import Prompts

path_portrait = "data/portrait"

input = Input()
prompts = Prompts()

dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(f"{path_portrait}/{dirname}")

for group in input.list_data:
    os.mkdir(f"{path_portrait}/{dirname}/{group[0]}")
    print("-" * os.get_terminal_size().columns)
    print(f"正在处理 group '{group[0]}':")
    print("-" * os.get_terminal_size().columns)

    portrait = prompts.initial_portrait

    for i, data in enumerate(group[1]):
        print(f"正在进行第{i+1}轮迭代")

        portrait = LLMCurrent.process(
            [
                {"role": "system", "content": prompts.prompt_iterate},
                {"role": "user", "content": data},
                {"role": "user", "content": portrait},
            ]
        )

        with open(f"{path_portrait}/{dirname}/{group[0]}/{i}.txt", "w", encoding="utf-8") as file:
            file.write(portrait)
