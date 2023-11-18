from accelerate import Accelerator


class AcceleratorSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AcceleratorSingleton, cls).__new__(cls)
            cls._instance._accelerator = Accelerator(
                log_with="wandb"
            )  # Private attribute
        return cls._instance

    @property
    def accelerator(self):
        return self._accelerator

    @accelerator.setter
    def accelerator(self, value):
        raise AttributeError("accelerator is read-only")

    def prepare(self, *args):
        return self._accelerator.prepare(*args)

    def print(self, *args, **kwargs):
        self._accelerator.print(*args, **kwargs)

    def log(self, values: dict):
        self._accelerator.log(values)

    def wait_for_everyone(self):
        self._accelerator.wait_for_everyone()

    def is_local_main_process(self):
        return self._accelerator.is_local_main_process()
