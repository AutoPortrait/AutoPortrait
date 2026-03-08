from typing import Optional
from typing import Protocol
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import warnings
import time
import re
import asyncio
import random


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
from typing import Optional

class APIKeyManager:
    def __init__(
        self,
        api_keys: list[str],
        cooldown_seconds: float = 60,
        min_request_interval: float = 20,
    ):
        self.api_keys = api_keys
        self.cooldown_seconds = cooldown_seconds
        self.min_request_interval = min_request_interval
        self.key_failure_times: dict[str, float] = {}
        self.key_last_request_times: dict[str, float] = {}

    def get_available_key(self) -> tuple[Optional[str], float]:
        now = time.time()
        available_keys = []
        min_wait_time = float('inf')
        
        for key in self.api_keys:
            failure_time = self.key_failure_times.get(key, 0)
            last_request_time = self.key_last_request_times.get(key, 0)
            
            failure_elapsed = now - failure_time
            request_elapsed = now - last_request_time
            
            if failure_elapsed >= self.cooldown_seconds and request_elapsed >= self.min_request_interval:
                available_keys.append(key)
            else:
                failure_wait = max(0, self.cooldown_seconds - failure_elapsed)
                request_wait = max(0, self.min_request_interval - request_elapsed)
                wait_time = max(failure_wait, request_wait)
                min_wait_time = min(min_wait_time, wait_time)
        
        if available_keys:
            return random.choice(available_keys), 0
        else:
            return None, min_wait_time

    def mark_key_request_sent(self, key: str):
        self.key_last_request_times[key] = time.time()

    def mark_key_failed(self, key: str):
        self.key_failure_times[key] = time.time()

    def mark_key_success(self, key: str):
        if key in self.key_failure_times:
            del self.key_failure_times[key]


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
        cooldown_seconds: float = 60,
        min_request_interval: float = 20,
    ):
        self.baseurl = base_url
        api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self.key_manager = APIKeyManager(api_keys, cooldown_seconds, min_request_interval)
        self.model_name = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_count = 0
        self.progress_log = progress_log
        self.logfile = None

    def _create_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key, base_url=self.baseurl)

    async def process(self, system_message: str, user_message: str) -> str:
        while True:
            api_key, wait_time = self.key_manager.get_available_key()
            
            if api_key is None:
                warnings.warn(f"所有 API Key 均在冷却中，等待 {wait_time:.1f} 秒...")
                await asyncio.sleep(wait_time)
                continue
            
            try:
                self.key_manager.mark_key_request_sent(api_key)
                client = self._create_client(api_key)
                start = time.time()
                if self.progress_log:
                    response = await client.chat.completions.create(
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
                            with open(self.progress_logfile, "a") as log:
                                log.write(content)
                            result += content
                        last = chunk
                    tokens = self.__get_usage_from_response(last)
                else:
                    response = await client.chat.completions.create(
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
                    tokens = self.__get_usage_from_response(response)
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
                result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
                self.key_manager.mark_key_success(api_key)
                return result
            except Exception as e:
                self.key_manager.mark_key_failed(api_key)
                warnings.warn(f"API请求失败 (Key: ...{api_key[-8:]}): {e}")
                remaining_key, _ = self.key_manager.get_available_key()
                if remaining_key:
                    warnings.warn(f"切换到其他可用的 API Key 重试...")
                    continue
                else:
                    _, wait_time = self.key_manager.get_available_key()
                    warnings.warn(f"所有 API Key 均不可用，等待 {wait_time:.1f} 秒后重试...")
                    await asyncio.sleep(wait_time)
                    continue

    def token_usage(self) -> int:
        return self.token_count

    def set_logfile(self, logfile: str) -> None:
        self.logfile = logfile

    def set_progress_logfile(self, logfile: str) -> None:
        self.progress_logfile = logfile
    
    def __get_usage_from_response(self, response: any) -> int:
        if response.usage and response.usage.total_tokens:
            return response.usage.total_tokens
        else:
            # warnings.warn(
            #     f"Warning: total_tokens is None in the response. Setting tokens to 0."
            # )
            # warnings.warn(f"Response: {response}")
            return 0


load_dotenv()
# LLMFast: LLMAbstract = LLMOpenAICompatible(
#     base_url="https://open.bigmodel.cn/api/paas/v4/",
#     api_key=os.getenv("ZHIPU_KEY"),
#     model_name="GLM-4-Plus",
#     top_p=0.7,
#     temperature=0.80,
#     max_tokens=4095,
# )

# LLMFast: LLMAbstract = LLMOpenAICompatible(
#     base_url="https://api.siliconflow.cn/v1",
#     api_key=os.getenv("SILICON_FLOW_API_KEY"),
#     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#     top_p=0.00,
#     temperature=0.00,
#     max_tokens=8192,
#     progress_log=True,
# )
# LLMInstructional: LLMAbstract = LLMOpenAICompatible(
#     base_url="https://api.siliconflow.cn/v1",
#     api_key=os.getenv("SILICON_FLOW_API_KEY"),
#     model_name="deepseek-ai/DeepSeek-R1",
#     # base_url="https://api.deepseek.com",
#     # api_key=os.getenv("DEEPSEEK_API_KEY"),
#     # model_name="deepseek-chat",
#     top_p=0.01,
#     temperature=0.00,
#     max_tokens=8192,
#     progress_log=True,
# )

LLMUniversal: LLMAbstract = LLMOpenAICompatible(
    base_url="https://api.scnet.cn/api/llm/v1",
    api_key=os.getenv("SCNET_API_KEY"),
    model_name="MiniMax-M2.5",
    # model_name="DeepSeek-V3.2",
    top_p=0.9,
    temperature=0.8,
    max_tokens=65536,
    progress_log=True,
)
LLMFast: LLMAbstract = LLMUniversal
LLMInstructional: LLMAbstract = LLMUniversal
