from abc import ABC, abstractmethod
from typing import List
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

class InputManager(ABC):
    @abstractmethod
    def extract_sentences_from_csv(self, csv_file: str) -> List[str]:
        """
        Extract sentences from a CSV file.
        
        :param csv_file: Path to the CSV file.
        :return: List of sentences extracted from the CSV file.
        """
        pass



class FAQInputManager(InputManager):
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
        loader = CSVLoader(file_path=csv_file, csv_args={"delimiter": ";"},encoding="utf-8-sig")
        data = loader.load()

        return data
       