"""Self-contained runtime files for converted Ming Tiny checkpoints."""

import copy
import json
import shutil
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE-APACHE-2.0",
    "WHISPER_LICENSE",
    "NOTICE",
    "NOTICE.txt",
)


def copy_supporting_files(source, destination):
    for name in TOKENIZER_FILES:
        path = Path(source) / name
        if path.is_file():
            target = Path(destination) / name
            if name == "tokenizer_config.json":
                config = json.loads(path.read_text())
                config.pop("auto_map", None)
                config["tokenizer_class"] = "PreTrainedTokenizerFast"
                target.write_text(
                    json.dumps(config, ensure_ascii=False, indent=2) + "\n"
                )
            else:
                shutil.copyfile(path, target)


def prepare_config(config, model_path):
    config = copy.deepcopy(config)
    config.pop("auto_map", None)
    config.pop("model_path", None)
    config.pop("talker_path", None)
    # Speculative MTP layers are not part of the autoregressive runtime.
    for key in ("text_config", "llm_config"):
        if key in config:
            config[key].pop("auto_map", None)
            config[key]["num_nextn_predict_layers"] = 0
    return config


def write_model_card(destination, source, model_id, config):
    from mlx_audio.tts.models.lingv3_tts.conversion import model_card

    model_card(
        destination,
        source,
        model_id,
        "Ming-3.0 ASR Tiny",
        "stt",
        'result = model.generate("speech.wav")\nprint(result.text)',
        "English and Chinese transcription. Input audio is resampled to 16 kHz. "
        "The audio encoder and MoE routers retain floating-point precision. "
        "Speculative MTP decoding and timestamps are not implemented.",
    )
