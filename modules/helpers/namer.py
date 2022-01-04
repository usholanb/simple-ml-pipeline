from typing import AnyStr, Dict
from utils.common import to_snake_case, is_outside_library


class Namer:

    @staticmethod
    def model_name(m_configs: Dict) -> AnyStr:
        name = m_configs.get("name")
        if is_outside_library(name):
            name = to_snake_case(name.split('.')[-1])
        return f'{name}_{m_configs.get("tag")}'

