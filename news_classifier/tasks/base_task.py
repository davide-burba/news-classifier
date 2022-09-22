from abc import ABC,abstractclassmethod

class BaseTask(ABC):
    def __init__(self,config,output_dir):
        self.config = config
        self.output_dir = output_dir

    @abstractclassmethod
    def run(self):
        pass