from typing import Sequence, Any

import tiktoken
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSequence
from langchain_openai import ChatOpenAI
from openai import BaseModel

from modules.rag.state import State, TokenUsageEntry


def invoke_model_and_receive_token_usage(
    model: ChatOpenAI,
    prompt: PromptValue,
    step: str,
) -> tuple:
    callback = UsageMetadataCallbackHandler()
    response = model.invoke(prompt, config={"callbacks": [callback]})

    model_name = model.first.bound.model_name
    usage = callback.usage_metadata.get(model_name, {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    token_usage: TokenUsageEntry = {
        "step_name": step,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    return response, token_usage
