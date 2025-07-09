
import tiktoken

from modules.rag.state import State


def get_token_usage(
        step_name: str,
        input_str: str,
        output_str: str,
        state: State,
        model_name: str = "gpt-4",
):
    """
    Counts tokens for input and output strings and stores them in state['token_usage'][step_name].
    """
    enc = tiktoken.encoding_for_model(model_name)
    input_tokens = len(enc.encode(input_str))
    output_tokens = len(enc.encode(output_str))

    old = state.get("token_usage",{"token_usage":{}})
    old[step_name]={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return {"token_usage":old}

