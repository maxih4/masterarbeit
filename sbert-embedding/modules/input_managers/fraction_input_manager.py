from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from module_instances import model_manager
from modules.input_managers.base_input_manager import BaseInputManager


class FractionInputManager(BaseInputManager):
    def __init__(self):
        """
        Initialize the FractionInputManager with a model manager.
        """
        self.model_manager = model_manager

    async def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Postprocess documents asynchronously by creating
        a normalized description of each fraction.
        """
        all_documents_to_store = []

        for document in documents:
            # robuste Extraktion mit fallback
            fraktion = document.metadata.get("Fraktion", "Unbekannt")
            allowed = document.metadata.get("Was darf rein", "")
            not_allowed = document.metadata.get("Was darf NICHT rein", "")

            page_content = (
                f"Die Fraktion {fraktion} darf folgendes enthalten: {allowed}. "
                f"Die Fraktion {fraktion} darf folgendes nicht enthalten: {not_allowed}."
            )

            new_doc = Document(page_content=page_content, metadata=document.metadata)

            all_documents_to_store.append(new_doc)

        return all_documents_to_store

    async def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract sentences from a CSV file containing fraction rules asynchronously.

        :param csv_file: Path to the CSV file.
        :return: List of sentences extracted from the CSV file.
        """
        loader = CSVLoader(
            file_path=csv_file,
            csv_args={"delimiter": ";"},
            encoding="utf-8-sig",
            metadata_columns=["Fraktion", "Was darf rein", "Was darf NICHT rein"],
            content_columns=[],
        )
        data = await loader.aload()
        return data
