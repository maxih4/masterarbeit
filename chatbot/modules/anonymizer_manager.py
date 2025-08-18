from langchain_experimental.data_anonymizer.base import AnonymizerBase


class AnonymizerManager:
    def __init__(self, anonymizer: AnonymizerBase):
        """
        Initialize the anonymizer manager.
        :param anonymizer: anonymizer instance
        """
        self.anonymizer = anonymizer
