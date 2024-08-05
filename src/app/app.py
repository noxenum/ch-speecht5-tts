import base64
import io
from typing import Optional, Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.io.wavfile import write

from src.app.translation_model_ct2 import TranslationModelCT2
from src.app.tts_model import TTSModel

app = FastAPI()

id_to_dialect = {0: "zh"}


class TTSInputModel(BaseModel):
    text_de: Optional[Union[str, list[str]]] = None
    text_ch: Optional[Union[str, list[str]]] = None
    voice_id: int


@app.get("/ping")
async def ping():
    return {"status_code": 200}


@app.post("/invocations", status_code=200)
async def predict(tts_input: TTSInputModel):
    dialect = id_to_dialect[tts_input.voice_id]

    if tts_input.text_de is not None:
        texts_ch = TranslationModelCT2.predict(
            dialect=dialect, text_de=tts_input.text_de, beam_size=1
        )
    else:
        texts_ch = tts_input.text_ch

        if isinstance(texts_ch, str):
            texts_ch = [texts_ch]

    wavs = []
    for text_ch in texts_ch:
        wav, sr = TTSModel.predict(speaker_id=tts_input.voice_id, text_ch=text_ch)
        wavs.append(wav)

    wav = np.concatenate(wavs)

    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sr, wav)
    wav_bytes = byte_io.read()

    audio_data = base64.b64encode(wav_bytes).decode("UTF-8")

    return {"status_code": 200, "audio": audio_data, "text_ch": " ".join(texts_ch)}
