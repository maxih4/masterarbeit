import asyncio
from module_instances import create_db_manager
from modules.input_managers.faq_input_manager import FAQInputManager
from modules.input_managers.fraction_input_manager import FractionInputManager
from modules.pipelines.base_csv_pipeline import BaseCSVPipeline
from utils.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)


async def main():
    db_manager = create_db_manager(drop_old=True)  # create inside async context
    pipelines = [
        BaseCSVPipeline(
            FAQInputManager(), csv_path="csv/faq.csv", db_manager=db_manager
        ),
        BaseCSVPipeline(
            FractionInputManager(), csv_path="csv/fraktionen.csv", db_manager=db_manager
        ),
    ]

    # await asyncio.gather(*(pipeline.run() for pipeline in pipelines))

    # First, create a list of coroutine objects by calling run() on each pipeline
    coroutines = [pipeline.run() for pipeline in pipelines]

    # Then, await all coroutines concurrently with asyncio.gather
    await asyncio.gather(*coroutines)

    for pipeline in pipelines:
        logger.info(f"Pipeline {pipeline.__class__.__name__} completed successfully.")

    logger.info("Finished.")


if __name__ == "__main__":
    asyncio.run(main())
