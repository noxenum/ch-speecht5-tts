import pickle

import numpy as np
import torch
from transformers import pipeline


class TTSModel:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    embeddings = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Started TTS model loading...")

            xvector_picke_path = "/opt/ml/model/xvector/embeddings.pickle"
            model_dir = "/opt/ml/model/st5"

            cls.model = pipeline("text-to-speech", model_dir)

            with open(xvector_picke_path, "rb") as f:
                cls.embeddings = pickle.load(f)

        return cls.model, cls.embeddings

    @classmethod
    def predict(cls, speaker_id: int, text_ch: str) -> tuple[np.ndarray, int]:
        model, embeddings = cls.get_model()

        spembs = torch.tensor(embeddings[speaker_id]).unsqueeze(0)

        with torch.no_grad():
            speech = model(text_ch, forward_params={"speaker_embeddings": spembs})

        return speech["audio"], speech["sampling_rate"]
