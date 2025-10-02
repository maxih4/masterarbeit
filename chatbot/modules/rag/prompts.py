from langchain_core.prompts import ChatPromptTemplate

# Standard base template
prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "{instructions}, correct examples: {positive_examples}, wrong examples (Do NOT do this): {negative_examples}",
        ),
        ("human", "{human_input_with_additional_information}"),
    ]
)
""" 
General prompt template.
:param instructions: system instructions
:param positive_examples: correct examples
:param negative_examples: incorrect examples
:param human_input_with_additional_information: user input with context
"""


# Template for one question
generate_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. "
            "Don't use more sentences than needed. You are only allowed to answer using the same language as the questions.",
        ),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)
"""
Prompt for answering a single question.
:param context: retrieved context
:param question: user question
"""


# Template for multiple sub-questions and final answer
generate_multiple_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following subquestions with the answer to it and answer the user input. "
            "Don't use any additional knowledge, only use the provided subquestions and their answers. If you don't know the answer, just say that you don't know. "
            "Use five sentences maximum and keep the answer concise. Don't use more sentences than needed. "
            "You are only allowed to answer using the same language as the questions.",
        ),
        ("human", "Userinput: {user_input}; sub-question->answer: {formatted_pairs}"),
    ]
)
"""
Prompt for combining multiple sub-questions into one final answer.
:param user_input: original user input
:param formatted_pairs: sub-questions and their answers
"""
