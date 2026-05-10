class Registor:
    _mapping = {}

    @classmethod
    def register(cls, name):
        def wrapper(target_cls):
            cls._mapping[name] = target_cls
            return target_cls
        return wrapper

    @classmethod
    def get(cls, name):
        return cls._mapping.get(name)