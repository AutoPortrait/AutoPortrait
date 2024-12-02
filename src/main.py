from datetime import datetime
import os
from LLM import LLMCurrent
from Input import Input
from Prompts import Prompts
from Classify import classify

path_portrait = "data/portrait"

input = Input()
prompts = Prompts()

dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(f"{path_portrait}/{dirname}")

portraits = list[tuple[str, int, str]]()  # group name, i, portrait

for group in input.list_data:
    os.mkdir(f"{path_portrait}/{dirname}/{group[0]}")
    print("-" * os.get_terminal_size().columns)
    print(f"正在处理'{group[0]}'组:")

    portraits.append([group[0], 0, prompts.initial_portrait])

    for data in group[1]:
        i = portraits[len(portraits) - 1][1]
        print(f"正在对'{group[0]}'组进行第{i+1}轮迭代")

        portrait = LLMCurrent.process(
            [
                {"role": "system", "content": prompts.prompt_iterate},
                {"role": "user", "content": data},
                {"role": "user", "content": portraits[len(portraits) - 1][2]},
            ]
        )

        with open(f"{path_portrait}/{dirname}/{group[0]}/{i}.txt", "w", encoding="utf-8") as file:
            file.write(portrait)
        portraits[len(portraits) - 1][1] += 1
        portraits[len(portraits) - 1][2] = portrait

for uncertain in input.uncertain_data:
    print("-" * os.get_terminal_size().columns)
    print(f"正在处理未分类访谈 '{uncertain[0]}':")

    classify_input_groups = [[group, portrait] for [group, _, portrait] in portraits]
    class_indexes = classify(classify_input_groups, uncertain[1])
    print(f"分类结果：{[input.list_data[i][0] for i in class_indexes]}")

    for index in class_indexes:
        i = portraits[index][1]
        group = input.list_data[index]
        print(f"正在对'{group[0]}'组进行第{i+1}轮迭代")

        portrait = LLMCurrent.process(
            [
                {"role": "system", "content": prompts.prompt_iterate},
                {"role": "user", "content": uncertain[1]},
                {"role": "user", "content": portraits[index][2]},
            ]
        )

        with open(
            f"{path_portrait}/{dirname}/{input.list_data[index][0]}/{i}.txt", "w", encoding="utf-8"
        ) as file:
            file.write(portrait)
        portraits[index][1] += 1
        portraits[index][2] = portrait
