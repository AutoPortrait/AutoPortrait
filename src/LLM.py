from typing import Protocol
import os
from dotenv import load_dotenv
from openai import OpenAI
import warnings
import time
import re


class LLMProcessError(Exception):
    def __init__(self, inner_exception: Exception):
        self.inner_exception = inner_exception


class LLMAbstract(Protocol):
    # May raise LLMProcessError
    def process(self, system_message: str, user_message: str) -> str: ...

    # May raise LLMProcessError
    def batch_process(
        self, system_message: str, user_message: str, result_size: int
    ) -> list[str]: ...

    def token_usage(self) -> int: ...

    def set_logfile(self, logfile: str) -> None: ...


import time
import warnings
import re
from openai import OpenAI


class LLMOpenAICompatible(LLMAbstract):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        top_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        console_log: bool = False,  # 新增参数
    ):
        self.baseurl = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_count = 0
        self.console_log = console_log  # 新增属性
        self.logfile = None

    def process(self, system_message: str, user_message: str) -> str:
        while True:
            try:
                start = time.time()
                if self.console_log:
                    # 启用流式响应
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        top_p=self.top_p,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=True,  # 启用流模式
                    )
                    result = ""
                    last = None
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            print(content, end="", flush=True)  # 实时显示生成的文本
                            result += content
                        last = chunk
                    tokens = last.usage.total_tokens
                    print()  # 换行
                else:
                    # 非流式响应
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        top_p=self.top_p,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,  # 关闭流模式
                    )
                    tokens = response.usage.total_tokens
                    result = response.choices[0].message.content
                end = time.time()

                self.token_count += tokens
                if self.logfile:
                    with open(self.logfile, "a") as log:
                        log.write(f"---- {time.ctime()} ----\n")
                        log.write(f"{self.model_name} ({self.baseurl})\n")
                        log.write(
                            f"{tokens} tokens, {end-start:.2f} s, {tokens/(end-start):.2f} tokens/s\n"
                        )
                        log.write(
                            f"top p: {self.top_p}, temperature: {self.temperature}, max tokens: {self.max_tokens}\n"
                        )
                        log.write(f"\n< system message >\n{system_message}\n")
                        log.write(f"\n< user message >\n{user_message}\n")
                        log.write(f"\n< response >\n{result}\n\n")

                # 移除 <think>...</think>
                result = re.sub(r"<think>.*?</think>", "", result)
                return result
            except Exception as e:
                warnings.warn(f"API请求失败，正在重试：{e}")
                time.sleep(60)
                continue

    def batch_process(self, system_message: str, user_message: str, result_size: int) -> list[str]:
        return [self.process(system_message, user_message) for _ in range(result_size)]

    def token_usage(self) -> int:
        return self.token_count

    def set_logfile(self, logfile: str) -> None:
        self.logfile = logfile


load_dotenv()
LLMCurrent: LLMAbstract = LLMOpenAICompatible(
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZHIPU_KEY"),
    model_name="GLM-4-Plus",
    top_p=0.7,
    temperature=0.80,
    max_tokens=4095,
)
LLMPrecise: LLMAbstract = LLMOpenAICompatible(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    model_name="deepseek-ai/DeepSeek-R1",
    # base_url="https://openrouter.ai/api/v1",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # model_name="deepseek/deepseek-r1:free",
    console_log=True,
)
