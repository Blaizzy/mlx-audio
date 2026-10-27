# Portions derived from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya
# Research, licensed under the Apache License 2.0 (see LICENSE in this directory).
"""Convert the official Maya-2 release to an mlx-audio model folder.

The release (maya-research/maya2-native) ships no top-level config.json, so the generic ``mlx_audio.convert`` cannot
read it directly; this builds one from ``maya3_config.json`` and ``backbone/config.json``, loads the checkpoint into
the MLX module tree (strictly), optionally quantizes the backbone, and writes weights, config, tokenizer and voices.

    python -m mlx_audio.tts.models.maya2.convert --hf-path maya-research/maya2-native --mlx-path maya2-native-bf16
    python -m mlx_audio.tts.models.maya2.convert --mlx-path maya2-native-8bit --q-bits 8
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx

BACKBONE_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "vocab_size",
    "num_key_value_heads",
    "max_position_embeddings",
    "head_dim",
    "tie_word_embeddings",
)


def source_dir(hf_path, revision=None) -> Path:
    p = Path(hf_path).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            hf_path,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "backbone/*",
                "voices/*",
                "LICENSE",
            ],
        )
    )


def build_config(src: Path) -> dict:
    release = json.loads((src / "maya3_config.json").read_text())
    if release.get("format") != "maya3-safetensors":
        raise ValueError(
            f"{src} is not a Maya-2 safetensors release (format={release.get('format')!r})"
        )
    bb = json.loads((src / "backbone" / "config.json").read_text())
    backbone = {k: bb[k] for k in BACKBONE_KEYS}
    backbone["rope_theta"] = float(
        (bb.get("rope_parameters") or {}).get("rope_theta") or bb["rope_theta"]
    )
    audio = release.get("audio", {})
    return {
        "model_type": "maya2",
        "backbone": backbone,
        "n_q": audio.get("n_codebooks", 32),
        "n_codes": audio.get("codebook_size", 2048),
        "sample_rate": audio.get("sample_rate", 24000),
        "languages": release.get("languages"),
        "source": "maya-research/maya2-native",
    }


def convert(
    hf_path="maya-research/maya2-native",
    mlx_path="maya2-native-mlx",
    q_bits=None,
    q_group_size=64,
    revision=None,
):
    from mlx.utils import tree_flatten

    from mlx_audio.lm.convert import quantize_model, save_config, save_model

    from .maya2 import Model, ModelConfig

    src = source_dir(hf_path, revision)
    config = build_config(src)
    model = Model(ModelConfig.from_dict(config))
    weights = {}
    for f in sorted(glob.glob(str(src / "*.safetensors"))):
        weights.update(mx.load(f))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"[maya2] loaded {n_params / 1e9:.2f}B parameters from {src}")
    if q_bits:
        model, config = quantize_model(
            model,
            config,
            q_group_size,
            q_bits,
            quant_predicate=model.model_quant_predicate,
        )
        print(
            f"[maya2] quantized the backbone projections to {q_bits} bits (group size {q_group_size})"
        )
    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    save_model(out, model, donate_model=True)
    save_config(config, config_path=out / "config.json")
    # the tokenizer goes in its own folder: next to a config.json without `transformers_version`, transformers flags
    # it as a Mistral tokenizer and suggests a regex "fix" that would change the tokenization
    (out / "tokenizer").mkdir(exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        if (src / "backbone" / name).exists():
            shutil.copy(src / "backbone" / name, out / "tokenizer" / name)
    shutil.copy(src / "voices" / "voices.json", out / "voices.json")
    if (src / "LICENSE").exists():
        shutil.copy(src / "LICENSE", out / "LICENSE")
    (out / "README.md").write_text(
        model_card(out.name, q_bits, config.get("languages"))
    )
    print(f"[maya2] wrote {out}")
    return out


def model_card(repo_name, q_bits, languages):
    variant = (
        f"{q_bits}-bit backbone projections (embeddings, depth decoder and heads unquantized)"
        if q_bits
        else "BF16, the reference numerics"
    )
    langs = "\n".join(f"  - {l}" for l in (languages or []))
    return f"""---
license: apache-2.0
library_name: mlx-audio
pipeline_tag: text-to-speech
base_model: maya-research/maya2-native
language:
{langs}
tags:
  - mlx
  - mlx-audio
  - text-to-speech
  - indic
  - voice-cloning
  - streaming
---

# {repo_name}

[maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native) converted for
[mlx-audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon: {variant}. All credit for the model goes to
[Maya Research](https://mayaresearch.ai).

Streaming text-to-speech for 11 Indic languages and English, with 20 voices and zero-shot cross-lingual voice cloning.

```bash
pip install -U mlx-audio
mlx_audio.tts.generate --model {repo_name} --text "வணக்கம்! நாளைக்கு 2 மணிக்கு meeting இருக்கு." --voice Kayal
```

```python
from mlx_audio.tts.utils import load_model

model = load_model("{repo_name}")
for result in model.generate("नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।", voice="Ritvik"):
    audio = result.audio  # 24 kHz
```

- **Language:** inferred from the text's script; `lang_code` picks between scripts' languages (hi / mr, bn / as).
- **Cloning:** `ref_audio`, `ref_text` and `ref_lang` clone a 3-10 s clip into any language. Get the speaker's
  permission first.
- **Streaming:** `stream=True`.
- **Text normalisation:** numbers, currency, dates and URLs are spoken correctly in every language
  (`normalize=True`, the default).

See the [mlx-audio docs page](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/maya2.md) for options,
voices and fidelity checks against the reference implementation.

Converted with `python -m mlx_audio.tts.models.maya2.convert{f' --q-bits {q_bits}' if q_bits else ''}`.

## License and attribution

The weights are Maya Research's, released under the **Apache License 2.0**; see `LICENSE` in this repo.

**Changes from the original:**
- converted from the release's safetensors to the mlx-audio layout (tensor names; tokenizer moved to `tokenizer/`)
{"- backbone projections quantized to " + str(q_bits) + " bits (affine, group size 64)" if q_bits else "- no numeric change (BF16, as released)"}

**Upstream components keep their own licenses:**
- **Qwen3-1.7B-Base** (Apache-2.0): the backbone the weights were fine-tuned from, included in these weights.
- **Kyutai Mimi** (CC-BY-4.0): the audio codec, downloaded from Kyutai at runtime and not redistributed here.
- **SpeechBrain ECAPA-TDNN** (Apache-2.0): the speaker encoder, downloaded at runtime and not redistributed here.

This conversion is not affiliated with or endorsed by Maya Research.

## Responsible use

This is Maya Research's guidance for the original model, and it applies equally here:
- **Clone only with permission.** Use voices you own or have explicit permission to use.
- **Disclose synthetic speech** to listeners, especially in calls and media.
- **No harmful use.** Never use it for deceptive impersonation, fraud, harassment, non-consensual content,
  disinformation, or to defeat voice authentication.
- **The built-in voices are synthetic personas**, not real people.
- **No watermark.** The model adds no audio watermark, so applications should add their own consent, disclosure and
  abuse safeguards.
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--hf-path",
        default="maya-research/maya2-native",
        help="release repo id or local directory",
    )
    ap.add_argument("--mlx-path", default="maya2-native-mlx", help="output directory")
    ap.add_argument(
        "--q-bits",
        type=int,
        default=None,
        help="quantize the backbone projections (8 recommended)",
    )
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    convert(a.hf_path, a.mlx_path, a.q_bits, a.q_group_size, a.revision)


if __name__ == "__main__":
    main()
