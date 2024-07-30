# ch-vits-tts
Inference server for Swiss German TTS based on the SpeechT5 model, compatible with Amazon SageMaker on AWS.

## Model folder structure
```
📦model
 ┣ 📂st5
 ┃ ┣ 📜added_tokens.json
 ┃ ┣ 📜config.json
 ┃ ┣ 📜preprocessor_config.json
 ┃ ┣ 📜pytorch_model.bin
 ┃ ┣ 📜special_tokens_map.json
 ┃ ┣ 📜spm_char.model
 ┃ ┗ 📜tokenizer_config.json
 ┣ 📂t5-ct2
 ┃ ┣ 📜config.json
 ┃ ┣ 📜model.bin
 ┃ ┗ 📜shared_vocabulary.json
 ┗ 📂xvector
   ┗ 📜embeddings.pickle
```

## Local instance
```
docker build . -t tts-st5
```

```
docker run --gpus=all -p 127.0.0.1:8080:8080 -v <PATH_TO_MODEL_FILES>\model:/opt/ml/model tts-st5
```