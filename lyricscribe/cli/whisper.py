from .model import Model
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch


class Whisper(Model):

    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name


    def load(self):
        self.processor = WhisperProcessor.from_pretrained(self.model_name)

        if (self.device == "cuda"):
            try:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    # with flash attention
                    self.model_name,
                    device_map = "cuda",
                    torch_dtype = torch.float16,
                     attn_implementation="flash_attention_2",
                )
            except Exception as e:
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
    
        self.model.eval()


    def name(self):
        return "Whisper"

    def transcribe(self, audio):
        if self.model is None or self.processor is None:
            raise RuntimeError("model not loaded")
        
        # assuming the proprocessing on the audio is all done

        input_features = self.processor(audio, sampling_rate = 16000, return_tensors = "pt").input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    
    