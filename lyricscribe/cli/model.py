from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, device = "cpu"):
        self.device =  device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        """Load the model"""

    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        """Run inference and return transcription"""

    @abstractmethod
    def name(self) -> str:
        """"Model name"""
