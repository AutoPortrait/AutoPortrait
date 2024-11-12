from datetime import datetime
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI, APIRequestFailedError

client: ZhipuAI

path_interview_directory = "data/interview"
path_interview_censored_directory = "data/interview/censored"
path_interview_index = "data/interview/index.txt"
path_initial_portrait = "prompt/初始人物画像.txt"
path_prompt_iterate = "prompt/迭代人物画像.txt"
path_portrait = "data/portrait"

list_data: list[str]
initial_portrait: str
prompt_iterate: str


def censor(text: str) -> str:
    if len(text) == 0:
        return text
    try:
        print(f"检查 {len(text)} 字 ... ", end="", flush=True)
        result = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": text},
            ],
            stream=False,
        )
        _ = result.choices[0].message.content
        print("合规")
        return text
    except APIRequestFailedError as e:
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


def initialize():
    global client
    load_dotenv()
    zhipu_key = os.getenv("ZHIPU_KEY")
    client = ZhipuAI(api_key=zhipu_key)

    global list_data, initial_portrait, prompt_iterate
    list_data = []
    with open(path_interview_index, "r", encoding="utf-8") as file:
        filenames = file.read().splitlines()
        for filename in filenames:
            censored_filename = f"{path_interview_censored_directory}/{filename}"
            if os.path.exists(censored_filename) == False:
                print(f"正在进行内容安全性预处理： {filename}")
                with open(f"{path_interview_directory}/{filename}", "r", encoding="utf-8") as file:
                    data = file.read()
                    censored_data = censor(data)
                with open(censored_filename, "w", encoding="utf-8") as file:
                    file.write(censored_data)
            with open(censored_filename, "r", encoding="utf-8") as file:
                list_data.append(file.read())
    with open(path_initial_portrait, "r", encoding="utf-8") as file:
        initial_portrait = file.read()
    with open(path_prompt_iterate, "r", encoding="utf-8") as file:
        prompt_iterate = file.read()


def iterate(data: str, portrait: str) -> str:
    result = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "system", "content": prompt_iterate},
            {"role": "user", "content": data},
            {"role": "user", "content": portrait},
        ],
        top_p=0.7,
        temperature=0.95,
        max_tokens=1024,
        stream=False,  # 关闭流模式，直接接收完整响
    )
    return result.choices[0].message.content


def main():
    initialize()
    portrait = initial_portrait
    dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(f"{path_portrait}/{dirname}")
    for i, data in enumerate(list_data):
        print()
        print("-" * os.get_terminal_size().columns)
        print(f"正在进行第{i+1}轮迭代 ...")

        portrait = iterate(data, portrait)
        with open(f"{path_portrait}/{dirname}/{i}.txt", "w", encoding="utf-8") as file:
            file.write(portrait)

        print("画像：")
        print(portrait)


if __name__ == "__main__":
    main()
