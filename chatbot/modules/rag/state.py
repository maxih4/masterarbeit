from operator import add
from typing import TypedDict, List, Dict, Annotated

from langchain_core.documents import Document


# Custom state for multiple parallel retrieve nodes
class RetrieveState(TypedDict):
    classifier: str
    question: str


# --- Merge/Reset-Reducer für Listen ---
def merge_or_reset(old: list, new: list):
    """
    Mergen oder Zurücksetzen einer Liste.
    - Wenn 'new' falsy ist ([], None), wird ein RESET ausgeführt -> [].
    - Andernfalls werden Elemente angehängt: old + new.
    Hinweis: Diese Semantik erlaubt explizites Leeren durch Übergabe von [].
    """
    if not new:  # [] oder None => reset
        return []
    return old + new  # normaler Merge


# --- Add/Reset-Reducer für Zähler ---
def add_or_reset(old: int, new: int):
    """
    Addieren oder Zurücksetzen eines Zählers.
    - Wenn 'new' falsy ist (0, None), wird RESET ausgeführt -> 0.
    - Andernfalls old + new.
    Achtung: 'new == 0' bedeutet hier bewusst 'reset' und nicht 'nichts addieren'.
    """
    if not new:  # 0 oder None => reset
        return 0
    return old + new  # normaler Add


# --- Mapping Frage -> Kontextdokumente ---
class QC(TypedDict):
    q: str  # gestellte Frage
    ctx: List["Document"]  # zugehörige Kontext-Dokumente (z. B. für Attribution)


# --- Tokenverbrauch pro Schritt/Knoten ---
class TokenUsageEntry(TypedDict):
    step_name: str  # Name des Graph-Schritts/Knotens
    input_tokens: int
    output_tokens: int


# --- Statusobjekt des Chatbots / Graph-States ---
class State(TypedDict):
    questions: List[str]  # Einzel Fragen des Users
    user_input: str  # letzte Nutzereingabe
    answer: str  # aktuelle Antwort
    last_answer: str  # vorherige Antwort
    last_user_questions: List[str]  # letzte Nutzerfragen (Kurzverlauf)
    classifier: str  # Klassifikation der Nutzeranfrage
    # Zähler werden addiert oder explizit zurückgesetzt (siehe add_or_reset)
    input_tokens: Annotated[int, add_or_reset]
    output_tokens: Annotated[int, add_or_reset]
    # Sammlungen werden gemerged oder explizit geleert (siehe merge_or_reset)
    token_usage: Annotated[List[TokenUsageEntry], merge_or_reset]
    qc_pairs: Annotated[List[QC], merge_or_reset]
