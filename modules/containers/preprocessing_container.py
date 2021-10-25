from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from modules.containers.container import Container
from modules.interfaces.csv_saver import CSVSaver


class PreprocessingContainer(Container):
    """ preprocessing container """
    csv_saver = CSVSaver()




