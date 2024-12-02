from typing import Protocol
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI, APIRequestFailedError

class LLMProcessError(Exception):
    def __init__(self, inner_exception: Exception):
        self.inner_exception = inner_exception


class LLMAbstract(Protocol):
    # May raise LLMProcessError
    def process(self, messages) -> str:
        ...

class LLMZhipuAI:
    def __init__(self):
        load_dotenv()
        zhipu_key = os.getenv("ZHIPU_KEY")
        self.client = ZhipuAI(api_key=zhipu_key)

    def process(self, messages) -> str:
        try:
            result = self.client.chat.completions.create(
                model="GLM-4-Air",
                messages=messages,
                top_p=0.7,
                temperature=0.95,
                max_tokens=1024,
                stream=False,  # 关闭流模式，直接接收完整响
            )
            return result.choices[0].message.content
        except APIRequestFailedError as e:
            raise LLMProcessError(e)

def LLMCurrent() -> LLMAbstract:
    return LLMZhipuAI()
