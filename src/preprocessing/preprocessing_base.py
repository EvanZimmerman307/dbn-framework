from typing import Dict, Any, Callable
from record import Record

class PreprocessingStep:
    """Abstract base class for preprocessing step"""
    name: str 
    def __init__(self, params: Dict[str, Any]): 
        self.params = params or {}
    def __call__(self, record: Record) -> Record: 
        raise NotImplementedError # step_instance(record)


STEP_REGISTRY: Dict[str, type[PreprocessingStep]] = {} # map preprocessing step names to steps

def register(name) -> Callable[[type], type]:
    """Decorator factory that returns a preprocessing step decorator""" 
    """When you use @register("some_name") on a step class, 
    it adds that step to the registry under the given name and returns the class unchanged"""
    """We pass name to the factory so it can register the preprocessing step"""
    
    def deco(cls: type) -> type:
        STEP_REGISTRY[name] = cls
        return cls
    
    return deco


