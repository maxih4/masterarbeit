import logging
from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.input_managers.base_input_manager import BaseInputManager

logger = logging.getLogger(__name__)


class QuestionsSchema(BaseModel):
    """
    Schema for generated questions from FAQ entries.
    :param questions: list of alternative user questions
    """

    questions: List[str] = Field(description="A list of questions.")


class FAQInputManager(BaseInputManager):
    """
    Input manager for FAQ data.
    Extracts FAQ entries from CSV and expands them into multiple question variants
    with the help of an LLM.
    """

    def __init__(self):
        """
        Initialize the FAQInputManager with a model manager.
        """
        self.model_manager = model_manager

    async def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Generate alternative user questions for each FAQ entry.
        :param documents: list of raw FAQ documents
        :return: list of original + generated documents
        """
        logger.info("[FAQInputManager] Starting postprocessing of FAQ entries.")

        structured_model = self.model_manager.llm_model.with_structured_output(
            schema=QuestionsSchema
        )

        all_documents_to_store: List[Document] = []

        for i, document in enumerate(documents, start=1):
            # Keep original document
            all_documents_to_store.append(document)

            antwort = document.metadata.get("Antwort", "")
            source = document.metadata.get("source")
            row = document.metadata.get("row")

            logger.debug(
                f"[FAQInputManager] Processing row {row}: Antwort={antwort[:40]}..."
            )

            # Ask LLM to generate alternative questions
            try:
                response = await structured_model.ainvoke(
                    f"""
                    Gegeben ist der folgende FAQ-Eintrag. Erzeuge eine Liste von **15 realistischen und unterschiedlichen Nutzerfragen**, 
                    die ein Kunde zu diesem Thema stellen könnte.  
                    Alle Fragen sollen semantisch zum FAQ-Eintrag passen, aber unterschiedliche Formulierungen, Synonyme oder Satzstellungen verwenden.  
                    Vermeide es, den FAQ-Eintrag wörtlich zu wiederholen, und berücksichtige sowohl formelle als auch informelle Varianten.  
                    Das Ziel ist es, die Trefferquote der semantischen Suche in einem RAG-System zu verbessern.

                    FAQ-Eintrag:
                    {document.page_content}
                    """
                )

                for question in response.questions:
                    new_doc = Document(
                        page_content=f"Frage: {question} Antwort: {antwort}",
                        metadata={"source": source, "row": row, "Antwort": antwort},
                    )
                    all_documents_to_store.append(new_doc)

                logger.debug(
                    f"[FAQInputManager] Generated {len(response.questions)} additional questions for row {row}."
                )
            except Exception as e:
                logger.warning(
                    f"[FAQInputManager] Failed to generate questions for row {row}: {e}"
                )

        logger.info(
            f"[FAQInputManager] Postprocessed {len(all_documents_to_store)} documents."
        )
        return all_documents_to_store

    async def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract FAQ entries from CSV asynchronously.
        :param csv_file: path to the CSV file
        :return: list of raw FAQ documents
        """
        logger.info(f"[FAQInputManager] Loading FAQ CSV file: {csv_file}")

        loader = CSVLoader(
            file_path=csv_file,
            csv_args={"delimiter": ";"},
            encoding="utf-8-sig",
            metadata_columns=["Antwort"],
            content_columns=["Frage", "Antwort"],
        )
        data = await loader.aload()

        logger.info(
            f"[FAQInputManager] Loaded {len(data)} FAQ entries from {csv_file}."
        )
        if data:
            logger.debug(
                f"[FAQInputManager] First entry content: {data[0].page_content[:80]}..."
            )
        return data
