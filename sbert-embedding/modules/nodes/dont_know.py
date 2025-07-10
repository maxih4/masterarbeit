from modules.rag.state import State


def dont_know(state: State):

    return {
        "answer": "Tut mir leid, dabei kann ich Ihnen nicht helfen. Bitte versuchen Sie es mit einer anderen Frage erneut"
    }
