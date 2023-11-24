from accelerate import Accelerator


class AcceleratorSingleton:
    _instance = None

    def __new__(cls, **kwargs):
        print(kwargs)
        if cls._instance is None:
            print("new")
            cls._instance = super(AcceleratorSingleton, cls).__new__(cls)
            cls._instance._accelerator = Accelerator(
                log_with="wandb", mixed_precision="fp16", **kwargs
            )
        return cls._instance

    @property
    def accelerator(self):
        return self._accelerator

    @accelerator.setter
    def accelerator(self, value):
        raise AttributeError("accelerator is read-only")

    @property
    def gradient_accumulation_steps(self):
        return self._accelerator.gradient_accumulation_steps

    @property
    def device(self):
        return self._accelerator.device

    @property
    def num_processes(self):
        return self._accelerator.num_processes

    def prepare(self, *args):
        return self._accelerator.prepare(*args)

    def print(self, *args, **kwargs):
        self._accelerator.print(*args, **kwargs)

    def log(self, values: dict):
        self._accelerator.log(values)

    def wait_for_everyone(self):
        self._accelerator.wait_for_everyone()

    def is_local_main_process(self):
        return self._accelerator.is_local_main_process

    def gather_for_metrics(self, input_data):
        return self._accelerator.gather_for_metrics(input_data)

    def accumulate(self, *models):
        return self._accelerator.accumulate(*models)

    def backward(self, loss, **kwargs):
        return self._accelerator.backward(loss, **kwargs)
