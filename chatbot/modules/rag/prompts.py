from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "{instructions}, correct examples: {positive_examples}, wrong examples(Do NOT do this): {negative_examples}",
        ),
        ("human", "{human_input_with_additional_information}"),
    ]
)


generate_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Don't use more sentences then you need. You are only allowed to answer using the same language as the questions.",
        ),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)


generate_multiple_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following subquestions with the answer to it and answer the user input. Don't use any additional knowledge, only use the provided sub questions and their answers. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise. Don't use more sentences then you need. You are only allowed to answer using the same language as the questions.",
        ),
        ("human", "Userinput: {user_input};sub-question->answer: {formatted_pairs}"),
    ]
)
