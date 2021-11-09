from typing import AnyStr

from utils.common import to_snake_case, is_outside_library


class Namer:

    @staticmethod
    def wrapper_name(m_configs) -> AnyStr:
        name = m_configs.get("name")
        if is_outside_library(name):
            name = to_snake_case(name.split('.')[-1])
        return f'{name}_{m_configs.get("tag")}'

    @staticmethod
    def dataset_name(configs) -> AnyStr:
        return configs.get("dataset").get('name')
