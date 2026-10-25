from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
import json
import tiktoken

# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    str(
                        item.get("text")
                        or item.get("content")
                        or item.get("value")
                        or ""
                    )
                )
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def cal_token(model: str, text: str):
    encoder = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoder.encode(_to_text(text), disallowed_special=()))
    return num_tokens


def cost_count(
    prompt,
    response,
    model_name,
    prompt_tokens=None,
    completion_tokens=None,
):
    prompt_len: int
    completion_len: int
    price: float

    prompt_len = (
        max(0, int(prompt_tokens))
        if prompt_tokens is not None
        else cal_token(model_name, prompt)
    )
    completion_len = (
        max(0, int(completion_tokens))
        if completion_tokens is not None
        else cal_token(model_name, response)
    )
    if model_name in MODEL_PRICE:
        prompt_price, completion_price = _resolve_model_price(model_name, prompt_len)
        price = (
            prompt_len * prompt_price / 1000000
            + completion_len * completion_price / 1000000
        )
    else:
        # Preserve token accounting for models without a monetary price.
        price = 0.0

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len


def _resolve_model_price(model_name, prompt_len):
    price_info = MODEL_PRICE[model_name]
    for tier in price_info.get("tiers", []):
        min_prompt_tokens = tier.get("min_prompt_tokens", 0)
        max_prompt_tokens = tier.get("max_prompt_tokens")
        if prompt_len >= min_prompt_tokens and (
            max_prompt_tokens is None or prompt_len <= max_prompt_tokens
        ):
            return tier["input"], tier["output"]
    return price_info["input"], price_info["output"]


MODEL_PRICE = {
    "gpt-3.5-turbo-0125": {"input": 0.5, "output": 1.5},
    "gpt-3.5-turbo-1106": {"input": 1.0, "output": 2.0},
    "gpt-4-1106-preview": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "gemini-1.5-flash-latest": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash-thinking-exp": {"input": 4.0, "output": 16.0},
    "llama-3.3-70b-versatile": {"input": 0.2, "output": 0.2},
    "Meta-Llama-3.1-70B-Instruct": {"input": 0.2, "output": 0.2},
    "llama-3.1-70b-instruct": {"input": 0.2, "output": 0.2},
    "deepseek-chat": {"input": 0.27, "output": 1.1},
    "deepseek-ai/DeepSeek-V3": {"input": 0.27, "output": 1.1},
    "qwen3.5-27b": {"input": 1.5, "output": 1.5},
    "qwen3.5-397b-a17b": {"input": 2.0, "output": 2.0},
    "deepseek/v4-flash": {"input": 0.8, "output": 0.8},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-flash-nothinking": {"input": 0.14, "output": 0.28},
    "deepseek-v4-flash-thinking": {"input": 0.14, "output": 0.28},
    "qwen3.5-flash": {
        "input": 0.029,
        "output": 0.287,
        "tiers": [
            {
                "min_prompt_tokens": 256000,
                "max_prompt_tokens": None,
                "input": 0.172,
                "output": 1.72,
            }
        ],
    },
    "gemini-2.5-flash-lite": {"input": 1.0, "output": 1.0},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "qwen3-8b": {"input": 1.0, "output": 1.0},
    "gemini-2.5-flash": {"input": 2.0, "output": 2.0},
    "gemini-2.5-flash-nothinking": {"input": 2.0, "output": 2.0},
    "gemini-2.5-flash-thinking": {"input": 2.0, "output": 2.0},
    "gpt-5.6-luna": {"input": 3.0, "output": 3.0},
    "deepseek-v4-pro": {"input": 3.0, "output": 3.0},
}
