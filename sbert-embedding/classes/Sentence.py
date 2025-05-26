from dataclasses import dataclass
from typing import Dict, List

from numpy import ndarray


@dataclass
class Sentence:
    """
    A class to represent a sentence with its text and embedding..
    """

    text: str
    sparse_vector: ndarray | List[Dict[str, float]] | List[ndarray]
    dense_vector: ndarray | List[Dict[str, float]] | List[ndarray]

