import time
from typing import Dict, List


def openai_call(client, messages: str | List[Dict[str, str]], max_tokens: int = 1024, model_name: str = "gpt-4o-mini") -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            response = client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise last_exception
            # Calculate delay with exponential backoff: base_delay * 2^attempt
            delay = 2**attempt
            time.sleep(delay)
    # This should never be reached due to the raise in the loop
    return ""
