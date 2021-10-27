from dependency_injector import containers, providers
from modules.helpers.csv_saver import CSVSaver


class SaverContainer(containers.DeclarativeContainer):
    csv_saver = providers.Singleton(
        CSVSaver
    )






