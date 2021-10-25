from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from modules.containers.container import Container
from utils.registry import registry


class TrainContainer(Container):
    """ preprocessing container """
    saver = providers.Singleton(
        registry.get_saver_class(Container.config.get('dataset').get('type', 'pkl'))
    )



