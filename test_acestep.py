import mlx.core as mx
from mlx_audio.tts.models.acestep.acestep import AceStepTTAModel
from mlx_audio.tts.models.acestep.config import AceStepConfig

def main():
    config = AceStepConfig()
    model = AceStepTTAModel(config)
    print("ACE-Step initialized successfully natively in MLX!")
    print(f"DiT: {model.dit}")
    print(f"Condition Encoder: {model.encoder}")

if __name__ == "__main__":
    main()
