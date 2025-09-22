import asyncio

from dotenv import load_dotenv

from module_instances import create_db_manager
from modules.input_managers.faq_input_manager import FAQInputManager
from modules.input_managers.fraction_input_manager import FractionInputManager
from modules.pipelines.base_csv_pipeline import BaseCSVPipeline
from utils.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)
load_dotenv()


async def main():
    # Instantiate db_manager and remove old data
    db_manager = create_db_manager(drop_old=True)
    # Create array with all pipelines that should get embedded
    pipelines = [
        BaseCSVPipeline(
            FAQInputManager(), csv_path="csv/faq.csv", db_manager=db_manager
        ),
        BaseCSVPipeline(
            FractionInputManager(), csv_path="csv/fraktionen.csv", db_manager=db_manager
        ),
    ]

    # First, create a list of coroutine objects by calling run() on each pipeline
    coroutines = [pipeline.run() for pipeline in pipelines]

    # Then, await all coroutines concurrently with asyncio.gather
    await asyncio.gather(*coroutines)

    logger.info("Finished.")


if __name__ == "__main__":
    asyncio.run(main())
