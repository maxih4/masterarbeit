from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "{instructions}, correct examples: {positive_example}, wrong examples(Do NOT do this): {negative_example}",
        ),
        ("human", "{human_input_with_additional_information}"),
    ]
)


generate_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the questions. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Don't use more sentences then you need. You are only allowed to answer using the same language as the questions.",
        ),
        ("human", "Context: {context}\n\nQuestions: {questions}"),
    ]
)
