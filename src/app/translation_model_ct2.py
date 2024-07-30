from typing import Union

import ctranslate2
import torch
from transformers.models.t5.tokenization_t5 import T5Tokenizer


class TranslationModelCT2:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tokenizer = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Started translation model loading...")

            model_path = "/opt/ml/model/t5-ct2"

            # float16 not working (outputs <pad> only)
            cls.model = ctranslate2.Translator(
                model_path, device=cls.device_name, compute_type="float32"
            )
            cls.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        return cls.model, cls.tokenizer

    @classmethod
    def predict(
        cls, dialect: str, text_de: Union[str, list[str]], beam_size: int = 1
    ) -> list[str]:
        model, tokenizer = cls.get_model()

        texts = text_de

        if isinstance(text_de, str):
            texts = [text_de]

        input_tokens = []

        for text in texts:
            input_tokens.append(
                tokenizer.convert_ids_to_tokens(tokenizer.encode(f"{dialect}: {text}"))
            )

        results = model.translate_batch(
            input_tokens,
            max_input_length=256,
            beam_size=beam_size,
            num_hypotheses=1,
        )

        preds = []
        for result in results:
            output_tokens = result.hypotheses[0]

            pred = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True
            )

            preds.append(pred)

        return preds
