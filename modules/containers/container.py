from dependency_injector import containers, providers
from utils.registry import registry


class Container(containers.DeclarativeContainer):
    """ Default Container """
    config = providers.Configuration()


