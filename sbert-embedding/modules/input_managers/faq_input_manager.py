from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from module_instances import model_manager
from modules.input_managers.base_input_manager import BaseInputManager


class FAQInputManager(BaseInputManager):
    def __init__(self):
        """
        Initialize the FAQInputManager
        """
        self.model_manager = model_manager

    async def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Postprocess documents asynchronously by generating additional user questions
        with the help of the LLM.
        """
        # LLM with structured output
        structured_model = self.model_manager.llm_model.with_structured_output(
            schema={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of questions.",
                    }
                },
                "required": ["questions"],
            }
        )

        all_documents_to_store = []

        for document in documents:
            # Keep original document
            all_documents_to_store.append(document)

            # store original document metadata
            antwort = document.metadata["Antwort"]
            source = document.metadata["source"]
            row = document.metadata["row"]

            # Generate possible questions asynchronously
            response = await structured_model.ainvoke(
                f"""
                Gegeben ist der folgende FAQ-Eintrag. Erzeuge eine Liste von **15 realistischen und unterschiedlichen Nutzerfragen**, die ein Kunde zu diesem Thema stellen könnte.  
                Alle Fragen sollen semantisch zum FAQ-Eintrag passen, aber unterschiedliche Formulierungen, Synonyme oder Satzstellungen verwenden.  
                Vermeide es, den FAQ-Eintrag wörtlich zu wiederholen, und berücksichtige sowohl formelle als auch informelle Varianten.  
                Das Ziel ist es, die Trefferquote der semantischen Suche in einem RAG-System zu verbessern.

                FAQ-Eintrag:
                {document.page_content}
                """
            )

            for question in response["questions"]:
                # Create a Document for each generated question
                new_doc = Document(
                    page_content=f"Frage: {question} Antwort: {antwort}",
                    metadata={"source": source, "row": row, "Antwort": antwort},
                )
                all_documents_to_store.append(new_doc)

        return all_documents_to_store

    async def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract sentences from a CSV file containing FAQs asynchronously.

        :param csv_file: Path to the CSV file.
        :return: List of sentences extracted from the CSV file.
        """
        loader = CSVLoader(
            file_path=csv_file,
            csv_args={"delimiter": ";"},
            encoding="utf-8-sig",
            metadata_columns=["Antwort"],
            content_columns=["Frage", "Antwort"],
        )
        data = await loader.aload()
        return data
