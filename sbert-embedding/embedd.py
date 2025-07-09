from module_instances import  create_db_manager
from modules.input_managers.faq_input_manager import FAQInputManager
from modules.input_managers.fraction_input_manager import FractionInputManager
from modules.pipelines.base_csv_pipeline import BaseCSVPipeline

from utils.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)


db_manager = create_db_manager(drop_old=True)

pipelines = [
    BaseCSVPipeline(FAQInputManager(), csv_path="csv/faq.csv", db_manager=db_manager),
    BaseCSVPipeline(FractionInputManager(), csv_path="csv/fraktionen.csv", db_manager=db_manager)
]



# Run pipelines
for pipeline in pipelines:
    pipeline.run()
    logger.info(f"Pipeline {pipeline.__class__.__name__} completed successfully.")
logger.info("Finished.")