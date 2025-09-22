from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI

from modules.rag.state import State, TokenUsageEntry


def invoke_model_and_receive_token_usage(
    model: ChatOpenAI,
    prompt: PromptValue,
    step: str,
) -> tuple:
    """
    Invoke a ChatOpenAI model and return its response with token usage.
    :param model: ChatOpenAI instance
    :param prompt: input prompt for the model
    :param step: name of the current processing step
    :return: tuple (response, token_usage)
    """
    # track token usage via callback
    callback = UsageMetadataCallbackHandler()

    # invoke model with prompt and attach callback
    response = model.invoke(prompt, config={"callbacks": [callback]})

    # extract usage stats from callback metadata
    model_name = model.first.bound.model_name
    usage = callback.usage_metadata.get(model_name, {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # build usage entry for logging/tracking
    token_usage: TokenUsageEntry = {
        "step_name": step,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    return response, token_usage
