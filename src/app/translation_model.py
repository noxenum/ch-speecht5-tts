import torch
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer


class TranslationModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tokenizer = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Started translation model loading...")

            model_path = "/opt/ml/model/t5"

            cls.model = T5ForConditionalGeneration.from_pretrained(model_path)
            cls.tokenizer = T5Tokenizer.from_pretrained("t5-small")

            cls.model.to(cls.device)

        return cls.model, cls.tokenizer

    @classmethod
    def predict(cls, dialect: str, text_de: str, beam_size: int = 1) -> str:
        model, tokenizer = cls.get_model()

        input_batch_pt = tokenizer(
            [f"{dialect}: {text_de}"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )

        decoded_out = model.generate(
            input_batch_pt["input_ids"].to(cls.device),
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=beam_size,
            num_return_sequences=1,
        )
        pred_batch = tokenizer.batch_decode(decoded_out, skip_special_tokens=True)
        return pred_batch[0]
