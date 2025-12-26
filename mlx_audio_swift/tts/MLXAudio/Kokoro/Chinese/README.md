# Chinese G2P for Kokoro TTS

Native Mandarin Chinese Grapheme-to-Phoneme (G2P) support for Kokoro TTS in Swift.

## Overview

This module provides a complete Chinese text-to-phoneme pipeline that converts Mandarin Chinese text to Bopomofo (注音符号) phonemes, which are then used by the Kokoro TTS model for speech synthesis.

Unlike the eSpeak-ng based approach (which uses Cantonese pronunciation), this module provides proper Mandarin Chinese phonemization with:

- **Jieba-style word segmentation** - Accurate tokenization of Chinese text
- **Pinyin conversion** - Character to pinyin mapping with polyphone disambiguation
- **Tone sandhi** - Mandarin tone modification rules (不/一 sandhi, third tone sandhi, neutral tone)
- **Bopomofo mapping** - Proper phoneme representation for the Kokoro model

## Requirements

### G2P Dictionaries

The Chinese G2P requires three dictionary files (gzip-compressed binary format):

1. **jieba.bin.gz** (~5MB) - Word segmentation dictionary
2. **pinyin_single.bin.gz** (~200KB) - Single character pinyin mappings
3. **pinyin_phrases.bin.gz** (~1MB, optional) - Phrase-level pinyin for polyphone disambiguation

These dictionaries can be downloaded from HuggingFace:
- Repository: `FluidInference/kokoro-82m-v1.1-zh-mlx`

## Usage

### Basic Usage

```swift
import MLXAudio

let tts = KokoroTTS()

// Initialize Chinese G2P with dictionary URLs
try tts.initializeChineseG2P(
    jiebaURL: jiebaURL,
    pinyinSingleURL: pinyinSingleURL,
    pinyinPhrasesURL: pinyinPhrasesURL  // Optional
)

// Generate audio with a Chinese voice
tts.generateAudio(voice: .zfXiaoxiao, text: "你好，世界！", speed: 1.0) { audio in
    // Handle audio chunk
}
```

### With Data (Embedded Dictionaries)

```swift
try tts.initializeChineseG2P(
    jiebaData: jiebaData,
    pinyinSingleData: pinyinSingleData,
    pinyinPhrasesData: pinyinPhrasesData  // Optional
)
```

### Using ChineseKokoroTokenizer Directly

```swift
let tokenizer = ChineseKokoroTokenizer()
try tokenizer.initialize(
    jiebaURL: jiebaURL,
    pinyinSingleURL: pinyinSingleURL
)

// Convert text to phonemes
let phonemes = try tokenizer.phonemize("你好世界")
// Result: "ㄋ阴3ㄏ外3十4ㄐ言4"

// Convert to token IDs
let tokenIds = tokenizer.tokenizeWithChineseVocab(phonemes)
```

## Chinese Voices

The following Kokoro voices support Chinese:

| Voice | Gender | Description |
|-------|--------|-------------|
| `zfXiaobei` | Female | Chinese female voice |
| `zfXiaoni` | Female | Chinese female voice |
| `zfXiaoxiao` | Female | Chinese female voice |
| `zfXiaoyi` | Female | Chinese female voice |
| `zmYunjian` | Male | Chinese male voice |
| `zmYunxi` | Male | Chinese male voice |
| `zmYunxia` | Male | Chinese male voice |
| `zmYunyang` | Male | Chinese male voice |

## Pipeline Overview

```
Chinese Text → Normalization → Word Segmentation → Pinyin Conversion
                                                         ↓
                                                   Tone Sandhi
                                                         ↓
                                                 Bopomofo Mapping
                                                         ↓
                                                   Token IDs
                                                         ↓
                                                  Kokoro Model
                                                         ↓
                                                   Audio Output
```

### Example Conversion

Input: `"你好世界"`

1. **Segmentation**: `["你好", "世界"]`
2. **Pinyin**: `["ni3", "hao3", "shi4", "jie4"]`
3. **Tone Sandhi**: `["ni2", "hao3", "shi4", "jie4"]` (third tone sandhi: 你好)
4. **Bopomofo**: `"ㄋ阴2ㄏ外3十4ㄐ言4"`

## Files

| File | Description |
|------|-------------|
| `ChineseG2P.swift` | Main G2P pipeline |
| `ChineseTokenizer.swift` | Jieba-style word segmentation |
| `PinyinConverter.swift` | Character to pinyin conversion |
| `ToneSandhi.swift` | Mandarin tone modification rules |
| `BopomofoMapper.swift` | Pinyin to Bopomofo mapping |
| `ChineseKokoroTokenizer.swift` | Integration with Kokoro TTS |

## Credits

This implementation is based on:
- [misaki](https://github.com/hexgrad/misaki) - Python G2P for Kokoro
- [pypinyin](https://github.com/mozillazg/python-pinyin) - Python pinyin library
- [jieba](https://github.com/fxsjy/jieba) - Chinese word segmentation
- [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) - Tone sandhi rules
