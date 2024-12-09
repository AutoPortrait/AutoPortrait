from typing import Protocol
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI, APIRequestFailedError


class LLMProcessError(Exception):
    def __init__(self, inner_exception: Exception):
        self.inner_exception = inner_exception


class LLMAbstract(Protocol):
    # May raise LLMProcessError
    def process(self, messages) -> str: ...

    # May raise LLMProcessError
    def batch_process(self, messages) -> list[str]: ...


class LLMZhipuAI:
    def __init__(self, model_name: str, top_p: float, temperature: float, max_tokens: int):
        load_dotenv()
        zhipu_key = os.getenv("ZHIPU_KEY")
        self.client = ZhipuAI(api_key=zhipu_key)
        self.model_name = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens

    def process(self, messages) -> str:
        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                top_p=self.top_p,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,  # 关闭流模式，直接接收完整响
            )
            return result.choices[0].message.content
        except APIRequestFailedError as e:
            raise LLMProcessError(e)


LLMCurrent: LLMAbstract = LLMZhipuAI("GLM-4-Air", 0.7, 0.80, 4095)
