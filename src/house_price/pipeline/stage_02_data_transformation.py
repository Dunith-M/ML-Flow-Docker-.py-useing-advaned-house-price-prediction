import logging
from ..components.data_transformation import DataTransformation
from ..config.configuration import ConfigurationManager
from ..utils.common import setup_logger

logger = logging.getLogger(__name__)


class Stage02DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logger.info("Stage 02: Data Transformation started")

            # Load configuration
            config = ConfigurationManager()
            data_config = config.get_data_transformation_config()

            # Initialize transformation class
            transformer = DataTransformation(data_config)

            # Run transformation step
            data = transformer.initiate_data_transformation()

            logger.info(
                "Preprocessor saved to: %s",
                data["preprocessor_path"],
            )

            logger.info("Stage 02: Data Transformation completed successfully")

            return data

        except Exception as e:
            logger.error(f"Error in Stage 02 pipeline: {e}")
            raise e


if __name__ == "__main__":
    setup_logger()

    try:
        pipeline = Stage02DataTransformationPipeline()
        data = pipeline.run()

        print("Data transformation completed successfully.")
        print("Preprocessor saved to:", data["preprocessor_path"])

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise e
