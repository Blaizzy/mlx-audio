import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert facebook/w2v-bert-2.0 to an MLX-compatible safetensors for IndexTTS2"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="facebook/w2v-bert-2.0",
        help="HuggingFace repo id",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision/commit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory containing config.json + w2vbert.safetensors",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import mlx.core as mx
    from safetensors import safe_open

    from mlx_audio.tts.models.indextts2.w2vbert import Wav2Vec2BertConfig, Wav2Vec2BertModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename="config.json",
            revision=args.revision,
        )
    )
    model_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename="model.safetensors",
            revision=args.revision,
        )
    )

    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = Wav2Vec2BertConfig(
        hidden_size=int(cfg_json["hidden_size"]),
        num_hidden_layers=int(cfg_json["num_hidden_layers"]),
        num_attention_heads=int(cfg_json["num_attention_heads"]),
        intermediate_size=int(cfg_json["intermediate_size"]),
        feature_projection_input_dim=int(cfg_json["feature_projection_input_dim"]),
        layer_norm_eps=float(cfg_json.get("layer_norm_eps", 1e-5)),
        position_embeddings_type=cfg_json.get("position_embeddings_type", "relative_key"),
        rotary_embedding_base=int(cfg_json.get("rotary_embedding_base", 10000)),
        max_source_positions=int(cfg_json.get("max_source_positions", 5000)),
        left_max_position_embeddings=int(cfg_json.get("left_max_position_embeddings", 64)),
        right_max_position_embeddings=int(cfg_json.get("right_max_position_embeddings", 8)),
        conv_depthwise_kernel_size=int(cfg_json.get("conv_depthwise_kernel_size", 31)),
        conformer_conv_dropout=float(cfg_json.get("conformer_conv_dropout", 0.1)),
        hidden_dropout=float(cfg_json.get("hidden_dropout", 0.0)),
        activation_dropout=float(cfg_json.get("activation_dropout", 0.0)),
        attention_dropout=float(cfg_json.get("attention_dropout", 0.0)),
        feat_proj_dropout=float(cfg_json.get("feat_proj_dropout", 0.0)),
    )

    model = Wav2Vec2BertModel(cfg)

    weights = {}
    # Use numpy framework to avoid requiring torch at conversion time.
    with safe_open(str(model_path), framework="numpy") as f:
        for k in f.keys():
            weights[k] = mx.array(f.get_tensor(k))

    weights = model.sanitize(weights)
    weights = {f"w2vbert.{k}": v for k, v in weights.items()}

    mx.save_safetensors(str(out_dir / "w2vbert.safetensors"), weights)

    cfg_out_path = out_dir / "config.json"
    if cfg_out_path.exists():
        root_cfg = json.loads(cfg_out_path.read_text(encoding="utf-8"))
    else:
        root_cfg = {"model_type": "indextts2", "sample_rate": 22050}

    root_cfg["w2vbert"] = {
        "hf_repo": args.hf_repo,
        "config": cfg_json,
        "weights": "w2vbert.safetensors",
        "prefix": "w2vbert.",
    }
    cfg_out_path.write_text(json.dumps(root_cfg, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 'w2vbert.safetensors'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
