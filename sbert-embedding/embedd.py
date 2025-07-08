import os
from module_instances import  db_manager
from modules.InputManager import FAQInputManager
from modules.pipelines.BaseCSVPipeline import BaseCSVPipeline

from utils.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)

pipelines = [
    BaseCSVPipeline(FAQInputManager(), csv_path="csv/faq.csv")]

# Clear existing data

db_manager.vector_store.delete(ids=None, expr="pk>0")
logger.info("Cleared existing data in the vector store.")

# Run pipelines
for pipeline in pipelines:
    pipeline.run()
    logger.info(f"Pipeline {pipeline.__class__.__name__} completed successfully.")
logger.info("Finished.")