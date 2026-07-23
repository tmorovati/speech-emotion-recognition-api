import librosa
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

DEFAULT_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
FEATURE_EXTRACTOR_MODEL = "facebook/wav2vec2-large-xlsr-53"
SAMPLE_RATE = 16000

_model = None
_feature_extractor = None


def _remap_state_dict_keys(state_dict):
    # This checkpoint was saved with an older classification-head naming
    # scheme (classifier.dense / classifier.output). Current transformers
    # expects projector / classifier, so a plain from_pretrained() silently
    # random-inits the head instead of loading the fine-tuned weights.
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace("classifier.dense.", "projector.")
        new_key = new_key.replace("classifier.output.", "classifier.")
        remapped[new_key] = value
    return remapped


def _load_model():
    global _model
    if _model is None:
        # The checkpoint's old classifier.dense layer was a 1024->1024 projector
        # (no dimensionality reduction), but Wav2Vec2Config defaults
        # classifier_proj_size to 256. Without this override, the remapped
        # weights below won't fit the model's projector/classifier shapes.
        config = Wav2Vec2Config.from_pretrained(DEFAULT_MODEL, classifier_proj_size=1024)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(DEFAULT_MODEL, config=config)
        weights_path = hf_hub_download(repo_id=DEFAULT_MODEL, filename="model.safetensors")
        state_dict = _remap_state_dict_keys(load_file(weights_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Weight remap left mismatches - missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        _model = model
    return _model


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(FEATURE_EXTRACTOR_MODEL)
    return _feature_extractor


def predict_emotion(audio_path):
    model = _load_model()
    feature_extractor = _get_feature_extractor()

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=-1)[0]

    id2label = model.config.id2label
    predictions = [
        {"label": id2label[i], "score": score.item()} for i, score in enumerate(scores)
    ]
    predictions.sort(key=lambda p: p["score"], reverse=True)
    return predictions
