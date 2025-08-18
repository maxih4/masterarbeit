from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseInputManager(ABC):
    @abstractmethod
    async def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract sentences from a CSV file asynchronously.

        :param csv_file: Path to the CSV file.
        :return: List of sentences extracted from the CSV file.
        """
        pass

    @abstractmethod
    async def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Postprocess the extracted documents asynchronously.

        :param documents: List of documents to postprocess.
        :return: List of postprocessed documents.
        """
        pass
