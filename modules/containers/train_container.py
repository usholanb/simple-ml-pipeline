from dependency_injector import containers, providers


class TrainContainer(containers.DeclarativeContainer):
    """ preprocessing container """
    config = providers.Configuration('config')



