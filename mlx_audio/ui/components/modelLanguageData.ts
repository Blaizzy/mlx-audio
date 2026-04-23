export const LANGUAGE_CODE: Record<string, string> = {
  Afrikaans: 'af', Albanian: 'sq', Amharic: 'am', Arabic: 'ar',
  Armenian: 'hy', Azerbaijani: 'az', Basque: 'eu', Belarusian: 'be',
  Bengali: 'bn', Bosnian: 'bs', Bulgarian: 'bg', Catalan: 'ca',
  Chinese: 'zh', Croatian: 'hr', Czech: 'cs', Danish: 'da',
  Dutch: 'nl', English: 'en', Estonian: 'et', Finnish: 'fi',
  French: 'fr', Galician: 'gl', Georgian: 'ka', German: 'de',
  Greek: 'el', Gujarati: 'gu', Hebrew: 'he', Hindi: 'hi',
  Hungarian: 'hu', Icelandic: 'is', Indonesian: 'id', Irish: 'ga',
  Italian: 'it', Japanese: 'ja', Javanese: 'jv', Kannada: 'kn',
  Kazakh: 'kk', Korean: 'ko', Latvian: 'lv', Lithuanian: 'lt',
  Macedonian: 'mk', Malay: 'ms', Malayalam: 'ml', Maltese: 'mt',
  Marathi: 'mr', Mongolian: 'mn', Nepali: 'ne', Norwegian: 'no',
  Persian: 'fa', Polish: 'pl', Portuguese: 'pt', Punjabi: 'pa',
  Romanian: 'ro', Russian: 'ru', Serbian: 'sr', Slovak: 'sk',
  Slovenian: 'sl', Somali: 'so', Spanish: 'es', Swahili: 'sw',
  Swedish: 'sv', Tamil: 'ta', Telugu: 'te', Thai: 'th',
  Turkish: 'tr', Ukrainian: 'uk', Urdu: 'ur', Uzbek: 'uz',
  Vietnamese: 'vi', Welsh: 'cy', Yoruba: 'yo', Zulu: 'zu',
};

export const MODEL_SUPPORTED_LANGS: Record<string, string[] | '*'> = {
  'mlx-community/whisper-large-v3-turbo-asr-fp16':    '*',
  'distil-whisper/distil-large-v3':                   ['en'],
  'UsefulSensors/moonshine-tiny':                     ['en'],
  'UsefulSensors/moonshine-base':                     ['en'],
  'facebook/mms-1b-fl102':                            '*',
  'facebook/mms-1b-all':                              '*',
  'fibm-granite/granite-4.0-1b-speech':               ['en', 'de', 'es', 'ja', 'pt'],
  'mlx-community/Voxtral-Mini-3B-2507-bf16':          ['en', 'de', 'es', 'fr', 'hi', 'ja', 'nl', 'pt'],
  'mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit': ['en', 'ar', 'de', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'nl', 'pt', 'ru', 'zh'],
  'mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16': ['en', 'ar', 'de', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'nl', 'pt', 'ru', 'zh'],
  'mlx-community/Qwen2-Audio-7B-Instruct-4bit':       '*',
  'mlx-community/Qwen3-ASR-1.7B-8bit':               ['en', 'ar', 'cs', 'da', 'de', 'el', 'es', 'fa', 'fi', 'fr', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'mk', 'ms', 'nl', 'pt', 'ro', 'ru', 'sv', 'th', 'vi', 'zh'],
  'mlx-community/Qwen3-ForcedAligner-0.6B-8bit':     ['en', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pt', 'ru', 'zh'],
  'mlx-community/parakeet-tdt-0.6b-v3':              ['en', 'bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fi', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'uk'],
  'mlx-community/VibeVoice-ASR-bf16':                '*',
};

export function getSupportedLanguageNames(modelValue: string): string[] | '*' {
  const codes = MODEL_SUPPORTED_LANGS[modelValue];
  if (!codes || codes === '*') return '*';
  return [
    'Detect',
    ...Object.entries(LANGUAGE_CODE)
      .filter(([, code]) => codes.includes(code))
      .map(([name]) => name),
  ];
}

export function getCompatibleModelValues(languageName: string): string[] | '*' {
  if (!languageName || languageName === 'Detect') return '*';
  const code = LANGUAGE_CODE[languageName];
  if (!code) return '*';
  return Object.entries(MODEL_SUPPORTED_LANGS)
    .filter(([, langs]) => langs === '*' || langs.includes(code))
    .map(([model]) => model);
}