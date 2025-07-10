from langchain_core.prompts import ChatPromptTemplate

first_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant for a recycling company's question-answering system. "
            "Your task is to rephrase the user's input into a clear, semantically meaningful, and complete question. "
            "Preserve exactly the meaning and scope of the original input — do not add or assume any information that is not explicitly present. "
            "Use the chat history only to resolve ambiguities, not to add unrelated topics. "
            "Output only the improved question, nothing else."
            "Do not add any words that are not needed to form a question. "
            "If the new input depends on the input before, rephrase the question that it can later be answered without knowing the history",
        ),
        (
            "human",
            "User Input: {user_input}\n\n" "Chat History (optional): {chat_history}",
        ),
    ]
)

second_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Don't use more sentences then you need. You are only allowed to answer using the same language as the question.",
        ),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)

classify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intent-classifier for a question-answering system. 
Your job is to classify the user input into exactly one of three categories:
- "internal_faq": questions about orders, payments, internal company processes or FAQ topics
- "waste_disposal_guidance": questions about what goes into which container or how to dispose of specific items
- "irrelevant_or_smalltalk": chit-chat, jokes, or unrelated questions
Only choose one of these three. Do not explain your choice.
        """,
        ),
        ("human", "User input: {user_input}"),
    ]
)
