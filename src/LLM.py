from typing import Protocol
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI, APIRequestFailedError
import warnings
import time


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


class LLMZhipuAI:
    def __init__(self, model_name: str, top_p: float, temperature: float, max_tokens: int):
        load_dotenv()
        zhipu_key = os.getenv("ZHIPU_KEY")
        self.client = ZhipuAI(api_key=zhipu_key)
        self.model_name = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_count = 0

    def process(self, system_message: str, user_message: str) -> str:
        while True:
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    top_p=self.top_p,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,  # 关闭流模式，直接接收完整响
                )
                self.token_count += result.usage.total_tokens
                return result.choices[0].message.content
            except APIRequestFailedError as e:
                if str(e).find("当前API请求过多") != -1:
                    warnings.warn("Too many API requests. Retry after 1 minute.")
                else:
                    warnings.warn(f"API request failed. Retry after 1 minute.\n{e}")
                time.sleep(60)
                continue

    def batch_process(self, system_message: str, user_message: str, result_size: int) -> list[str]:
        # ZhipuAI does not support batch processing
        return [self.process(system_message, user_message) for _ in range(result_size)]

    def token_usage(self) -> int:
        return self.token_count


LLMCurrent: LLMAbstract = LLMZhipuAI("GLM-4-Air", 0.7, 0.80, 4095)
