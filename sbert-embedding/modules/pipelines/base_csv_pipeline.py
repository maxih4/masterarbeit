from abc import ABC
import logging
from module_instances import model_manager
from modules.input_managers.base_input_manager import BaseInputManager

logger = logging.getLogger(__name__)


class BaseCSVPipeline(ABC):
    def __init__(self, input_manager, csv_path, db_manager):
        self.db_manager = db_manager
        self.model_manager = model_manager
        self.input_manager: BaseInputManager = input_manager
        self.csv_path: str = csv_path

    async def run(self):
        logger.info(f"Running pipeline: {self.__class__.__name__}")
        path = self.csv_path
        input_manager = self.input_manager

        logger.info(f"Reading file: {path}")
        documents = await input_manager.extract_sentences_from_csv(path)
        logger.info(f"Extracted {len(documents)} sentences from CSV file.")

        documents = await input_manager.postprocess_documents(documents)
        logger.info(f"Postprocessed {len(documents)} sentences.")

        # Use async add_documents call here
        await self.db_manager.vector_store.aadd_documents(documents)

        logger.info(f"Stored {len(documents)} documents in the vector store.")
        logger.info(f"Pipeline {self.__class__.__name__} done")
