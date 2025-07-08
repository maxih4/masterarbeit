from abc import ABC, abstractmethod
from typing import List
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

from module_instances import model_manager


class InputManager(ABC):
    @abstractmethod
    def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract sentences from a CSV file.
        
        :param csv_file: Path to the CSV file.
        :return: List of sentences extracted from the CSV file.
        """
        pass

    @abstractmethod
    def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Postprocess the extracted documents.

        :param documents: List of documents to postprocess.
        :return: List of postprocessed documents.
        """
        pass


class FAQInputManager(InputManager):
    def __init__(self):
        """
        Initialize the FAQInputManager with a model manager.

        :param model_manager: The model manager to be used for generating questions.
        """
        self.model_manager = model_manager


    def postprocess_documents(self, documents: List[Document]) -> List[Document]:
        # LLM with structured output
        structured_model = self.model_manager.llm_model.with_structured_output(schema={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of questions."
                }
            },
            "required": ["questions"]
        })

        all_documents_to_store = []

        for document in documents:
            # Keep original document
            all_documents_to_store.append(document)

            # Generate possible questions
            response = structured_model.invoke(
                f"""
                Gegeben ist der folgende FAQ-Eintrag. Erzeuge eine Liste von **15 realistischen und unterschiedlichen Nutzerfragen**, die ein Kunde zu diesem Thema stellen könnte.  
                Alle Fragen sollen semantisch zum FAQ-Eintrag passen, aber unterschiedliche Formulierungen, Synonyme oder Satzstellungen verwenden.  
                Vermeide es, den FAQ-Eintrag wörtlich zu wiederholen, und berücksichtige sowohl formelle als auch informelle Varianten.  
                Das Ziel ist es, die Trefferquote der semantischen Suche in einem RAG-System zu verbessern.

                FAQ-Eintrag:
                {document.page_content}
                """
            )
            for question in response['questions']:
                # Create a Document for each generated question
                new_doc = Document(
                    page_content="Frage: " + question + " Antwort: " + document.metadata["Antwort"],
                    metadata={"source": document.metadata["source"], "row": document.metadata["row"]}
                )
                all_documents_to_store.append(new_doc)
        return all_documents_to_store

    def extract_sentences_from_csv(self, csv_file: str) -> List[Document]:
        """
        Extract sentences from a CSV file containing FAQs.
        
        :param csv_file: Path to the CSV file.
        :return: List of sentences extracted from the CSV file.
        """
        # import csv

        # sentences = []
        # with open(csv_file, newline='', encoding='utf-8') as file:
        #     reader = csv.reader(file, delimiter=';')
        #     next(reader, None)  # Skip the header row
        #     for row in reader:
        #         if row:  # Make sure the row is not empty
        #              sentences.append(row[0] + " : " + row[1])  # Assuming the first column is the question and the second is the answer
        # return sentences

        loader = CSVLoader(file_path=csv_file, csv_args={"delimiter": ";"},encoding="utf-8-sig",metadata_columns=["Antwort"], content_columns=["Frage","Antwort"])
        data = loader.load()

        return data
       