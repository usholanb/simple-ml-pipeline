from dependency_injector import containers, providers
from modules.helpers.csv_saver import CSVSaver


class ConfigContainer(containers.DeclarativeContainer):
    """ preprocessing container """
    config = providers.Configuration('config')


class SaverContainer(containers.DeclarativeContainer):
    csv_saver = providers.Singleton(
        CSVSaver, ConfigContainer.config
    )






