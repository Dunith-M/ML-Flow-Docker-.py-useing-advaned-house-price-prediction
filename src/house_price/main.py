from .config.configuration import ConfigurationManager


def main() -> None:
    config = ConfigurationManager()

    data_config = config.get_data_ingestion_config()
    model_config = config.get_model_trainer_config()

    print(data_config)
    print(model_config)


if __name__ == "__main__":
    main()
