from typing import Protocol
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import warnings
import time
import re
import asyncio


class LLMProcessError(Exception):
    def __init__(self, inner_exception: Exception):
        self.inner_exception = inner_exception


class LLMAbstract(Protocol):
    # May raise LLMProcessError
    async def process(self, system_message: str, user_message: str) -> str: ...

    def token_usage(self) -> int: ...

    def set_logfile(self, logfile: str) -> None: ...

    def set_progress_logfile(self, logfile: str) -> None: ...


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
        progress_log: bool = False,
    ):
        self.baseurl = base_url
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_count = 0
        self.progress_log = progress_log
        self.logfile = None

    async def process(self, system_message: str, user_message: str) -> str:
        while True:
            try:
                start = time.time()
                if self.progress_log:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        top_p=self.top_p,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=True,
                    )
                    result = ""
                    last = None
                    async for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            # print(content, end="", flush=True)
                            with open(self.progress_logfile, "a") as log:
                                log.write(content)
                            result += content
                        last = chunk
                    tokens = last.usage.total_tokens
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        top_p=self.top_p,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,
                    )
                    tokens = response.usage.total_tokens
                    result = response.choices[0].message.content
                end = time.time()

                self.token_count += tokens
                result = result.strip()
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

                result: str = result
                result = re.sub(r"<think>.*?</think>", "", result).strip()
                return result
            except Exception as e:
                warnings.warn(f"API请求失败，正在重试：{e}")
                await asyncio.sleep(60)
                continue

    def token_usage(self) -> int:
        return self.token_count

    def set_logfile(self, logfile: str) -> None:
        self.logfile = logfile

    def set_progress_logfile(self, logfile: str) -> None:
        self.progress_logfile = logfile


load_dotenv()
# LLMFast: LLMAbstract = LLMOpenAICompatible(
#     base_url="https://open.bigmodel.cn/api/paas/v4/",
#     api_key=os.getenv("ZHIPU_KEY"),
#     model_name="GLM-4-Plus",
#     top_p=0.7,
#     temperature=0.80,
#     max_tokens=4095,
# )
LLMFast: LLMAbstract = LLMOpenAICompatible(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    top_p=0.00,
    temperature=0.00,
    max_tokens=8192,
    progress_log=True,
)
LLMInstructional: LLMAbstract = LLMOpenAICompatible(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    model_name="deepseek-ai/DeepSeek-R1",
    top_p=0.00,
    temperature=0.00,
    max_tokens=8192,
    progress_log=True,
)
