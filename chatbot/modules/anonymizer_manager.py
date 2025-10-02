from langchain_experimental.data_anonymizer import PresidioAnonymizer


class AnonymizerManager:
    def __init__(self, anonymizer: PresidioAnonymizer):
        """
        Initialize the anonymizer manager.
        :param anonymizer: anonymizer instance
        """
        self.anonymizer = anonymizer
