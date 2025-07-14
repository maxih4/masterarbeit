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
Your job is to classify the user input into exactly one of these four categories:

- "internal_faq":  
  The user asks for general information about the company, internal processes, payments, invoices, contact details, or FAQ topics.

- "waste_disposal_guidance":  
  The user asks for general information about how to correctly dispose of a material, what goes into which bin/container, or what type of container is appropriate for specific waste.  
  This includes questions like "What kind of container do I need for earth and gravel?".  
  It does NOT include asking to order, schedule, pick up, or remove a container.

- "irrelevant_or_smalltalk":  
  Off-topic questions, chit-chat, jokes, greetings, or anything unrelated to the recycling company.

- "complex_query_customer_support":  
  The user requests a specific service action or has a complaint.  
  Examples include asking to order, schedule, pick up, deliver, or remove a container at a specific address, asking why something wasn’t done, when something will be delivered or collected, or reporting a problem.

Always focus on the *intent of the user*:  
- If the user only asks for information or advice (e.g., how-to, what type), choose `waste_disposal_guidance`.  
- If the user asks for an action to be performed (e.g., pickup, delivery, order, complaint), choose `complex_query_customer_support`.

Only choose one of these four categories. Do not explain your choice.

Examples:

User: Wie entsorge ich eine alte Matratze?  
Classification: waste_disposal_guidance

User: Wann wird meine Tonne geleert?  
Classification: complex_query_customer_support

User: Können Sie die Papiermülltonne in der Musterstraße 123 abholen?  
Classification: complex_query_customer_support

User: Ich möchte wissen, was in den Gelben Sack darf.  
Classification: waste_disposal_guidance

User: Wie bekomme ich meine Rechnung?  
Classification: internal_faq

User: Hallo, wie geht’s?  
Classification: irrelevant_or_smalltalk

User: Meine Container wurden diese Woche nicht geleert, warum?  
Classification: complex_query_customer_support

User: Ich brauche einen Container für Bodenaushub. Materialien sind Erde und Kies. Was für einen Container brauche ich?  
Classification: waste_disposal_guidance

User: Bitte liefern Sie mir einen Container für Erde und Kies an die Musterstraße 123.  
Classification: complex_query_customer_support
            """,
        ),
        ("human", "User input: {user_input}"),
    ]
)
