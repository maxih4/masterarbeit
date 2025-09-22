import logging
from module_instances import model_manager
from modules.input_managers.base_input_manager import BaseInputManager

logger = logging.getLogger(__name__)


class BaseCSVPipeline:
    """
    Base pipeline for CSV ingestion and embedding.
    :param input_manager: instance implementing CSV extraction & postprocessing
    :param csv_path: path to the CSV file
    :param db_manager: manager providing a vector_store with aadd_documents
    """

    def __init__(self, input_manager: BaseInputManager, csv_path: str, db_manager):
        self.db_manager = db_manager
        self.model_manager = model_manager
        self.input_manager: BaseInputManager = input_manager
        self.csv_path: str = csv_path

    async def run(self) -> None:
        """
        Execute the pipeline: read CSV, postprocess, and store embeddings.
        :return: None
        """
        logger.info(f"Running pipeline: {self.__class__.__name__}")
        path = self.csv_path
        input_manager = self.input_manager

        # 1) Read and split into sentences/documents
        logger.info(f"Reading file: {path}")
        documents = await input_manager.extract_sentences_from_csv(path)
        logger.info(f"Extracted {len(documents)} sentences from CSV file.")

        # 2) Postprocess (cleanup, metadata, chunking, etc.)
        documents = await input_manager.postprocess_documents(documents)
        logger.info(f"Postprocessed {len(documents)} sentences.")

        # 3) Store in vector store (async)
        await self.db_manager.vector_store.aadd_documents(documents)

        logger.info(f"Stored {len(documents)} documents in the vector store.")
        logger.info(f"Pipeline {self.__class__.__name__} done")
