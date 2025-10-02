import logging
from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from module_instances import model_manager
from modules.input_managers.base_input_manager import BaseInputManager

logger = logging.getLogger(__name__)


class FractionInputManager(BaseInputManager):
    """
    Input manager for fraction rules (waste disposal fractions).
    Handles CSV extraction and postprocessing of fraction rules.
    """

    def __init__(self):
        """
        Initialize the FractionInputManager with a model manager.
        """
        self.model_manager = model_manager

    async def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Normalize extracted documents into clean fraction descriptions.
        :param documents: list of raw documents from CSV
        :return: list of processed documents
        """
        logger.info("[FractionInputManager] Starting postprocessing.")
        all_documents_to_store: List[Document] = []

        for i, document in enumerate(documents, start=1):
            fraktion = document.metadata.get("Fraktion", "Unbekannt")
            allowed = document.metadata.get("Was darf rein", "")
            not_allowed = document.metadata.get("Was darf NICHT rein", "")
            source = document.metadata.get("source")
            row = document.metadata.get("row")

            page_content = (
                f"Die Fraktion {fraktion} darf folgendes enthalten: {allowed}. "
                f"Die Fraktion {fraktion} darf folgendes nicht enthalten: {not_allowed}."
            )

            new_doc = Document(
                page_content=page_content,
                metadata={
                    "source": source,
                    "row": row,
                    "Fraktion": fraktion,
                    "Was darf rein": allowed,
                    "Was darf NICHT rein": not_allowed,
                },
            )

            all_documents_to_store.append(new_doc)

            logger.debug(
                f"[FractionInputManager] Processed row {row}: Fraktion={fraktion}, "
                f"Allowed={allowed[:30]}..., NotAllowed={not_allowed[:30]}..."
            )

        logger.info(
            f"[FractionInputManager] Postprocessed {len(all_documents_to_store)} documents."
        )
        return all_documents_to_store

    async def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract fraction rules from CSV file asynchronously.
        :param csv_file: path to the CSV file
        :return: list of raw documents
        """
        logger.info(f"[FractionInputManager] Loading CSV file: {csv_file}")

        loader = CSVLoader(
            file_path=csv_file,
            csv_args={"delimiter": ";"},
            encoding="utf-8-sig",
            metadata_columns=["Fraktion", "Was darf rein", "Was darf NICHT rein"],
            content_columns=[],
        )
        data = await loader.aload()

        logger.info(f"[FractionInputManager] Loaded {len(data)} rows from {csv_file}.")
        if data:
            logger.debug(
                f"[FractionInputManager] First row metadata: {data[0].metadata}"
            )
        return data
