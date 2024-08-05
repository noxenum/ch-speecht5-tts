import pickle

import numpy as np
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor


class TTSModel:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    processor = None
    model = None
    vocoder = None
    embeddings = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Started TTS model loading...")

            xvector_picke_path = "/opt/ml/model/xvector/embeddings.pickle"
            model_dir = "/opt/ml/model/st5"
            vocoder_dir = "/opt/ml/model/vocoder"

            cls.processor = SpeechT5Processor.from_pretrained(model_dir)
            cls.model = SpeechT5ForTextToSpeech.from_pretrained(model_dir)
            cls.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_dir)

            with open(xvector_picke_path, "rb") as f:
                cls.embeddings = pickle.load(f)

        return cls.processor, cls.model, cls.vocoder, cls.embeddings

    @classmethod
    def predict(cls, speaker_id: int, text_ch: str) -> tuple[np.ndarray, int]:
        processor, model, vocoder, embeddings = cls.get_model()

        spembs = torch.tensor(embeddings[speaker_id]).unsqueeze(0)

        with torch.no_grad():
            inputs = processor(text=text_ch, return_tensors="pt")

            speech = model.generate_speech(inputs["input_ids"], spembs, vocoder=vocoder)

        return speech.numpy(), 16000
