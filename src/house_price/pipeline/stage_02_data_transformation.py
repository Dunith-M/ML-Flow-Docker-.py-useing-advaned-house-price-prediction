from house_price.config.configuration import ConfigurationManager
from house_price.components.data_transformation import DataTransformation
from house_price.utils.common import setup_logger

import logging

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

            # Debug outputs (you can remove later)
            logger.info(f"Numerical Features: {data['num_features']}")
            logger.info(f"Categorical Features: {data['cat_features']}")

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

        # Optional print (for development only)
        print("Numerical Features:", data["num_features"])
        print("Categorical Features:", data["cat_features"])

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise e