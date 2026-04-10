// Language full name → ISO code
export const LANGUAGE_CODE: Record<string, string> = {
  Afrikaans: 'AF', Albanian: 'SQ', Amharic: 'AM', Arabic: 'AR',
  Armenian: 'HY', Azerbaijani: 'AZ', Basque: 'EU', Belarusian: 'BE',
  Bengali: 'BN', Bosnian: 'BS', Bulgarian: 'BG', Catalan: 'CA',
  Chinese: 'ZH', Croatian: 'HR', Czech: 'CS', Danish: 'DA',
  Dutch: 'NL', English: 'EN', Estonian: 'ET', Finnish: 'FI',
  French: 'FR', Galician: 'GL', Georgian: 'KA', German: 'DE',
  Greek: 'EL', Gujarati: 'GU', Hebrew: 'HE', Hindi: 'HI',
  Hungarian: 'HU', Icelandic: 'IS', Indonesian: 'ID', Irish: 'GA',
  Italian: 'IT', Japanese: 'JA', Javanese: 'JV', Kannada: 'KN',
  Kazakh: 'KK', Korean: 'KO', Latvian: 'LV', Lithuanian: 'LT',
  Macedonian: 'MK', Malay: 'MS', Malayalam: 'ML', Maltese: 'MT',
  Marathi: 'MR', Mongolian: 'MN', Nepali: 'NE', Norwegian: 'NO',
  Persian: 'FA', Polish: 'PL', Portuguese: 'PT', Punjabi: 'PA',
  Romanian: 'RO', Russian: 'RU', Serbian: 'SR', Slovak: 'SK',
  Slovenian: 'SL', Somali: 'SO', Spanish: 'ES', Swahili: 'SW',
  Swedish: 'SV', Tamil: 'TA', Telugu: 'TE', Thai: 'TH',
  Turkish: 'TR', Ukrainian: 'UK', Urdu: 'UR', Uzbek: 'UZ',
  Vietnamese: 'VI', Welsh: 'CY', Yoruba: 'YO', Zulu: 'ZU',
};

// '*' = supports all languages in the selector
export const MODEL_SUPPORTED_LANGS: Record<string, string[] | '*'> = {
  'mlx-community/whisper-large-v3-turbo-asr-fp16':    '*',
  'distil-whisper/distil-large-v3':                   ['EN'],
  'UsefulSensors/moonshine-tiny':                     ['EN'],
  'UsefulSensors/moonshine-base':                     ['EN'],
  'facebook/mms-1b-fl102':                            '*',
  'facebook/mms-1b-all':                              '*',
  'fibm-granite/granite-4.0-1b-speech':               ['EN', 'DE', 'ES', 'JA', 'PT'],
  'mlx-community/Voxtral-Mini-3B-2507-bf16':          ['EN', 'DE', 'ES', 'HI', 'JA', 'PT'],
  'mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit': ['EN', 'AR', 'DE', 'ES', 'FR', 'HI', 'IT', 'JA', 'KO', 'NL', 'PT', 'RU', 'ZH'],
  'mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16': ['EN', 'AR', 'DE', 'ES', 'FR', 'HI', 'IT', 'JA', 'KO', 'NL', 'PT', 'RU', 'ZH'],
  'mlx-community/Qwen2-Audio-7B-Instruct-4bit':       '*',
  'mlx-community/Qwen3-ASR-1.7B-8bit':               ['EN', 'AR', 'CS', 'DA', 'DE', 'EL', 'ES', 'FA', 'FI', 'FR', 'HI', 'HU', 'ID', 'IT', 'JA', 'KO', 'MK', 'MS', 'NL', 'PT', 'RO', 'RU', 'SV', 'TH', 'VI', 'ZH'],
  'mlx-community/Qwen3-ForcedAligner-0.6B-8bit':     ['EN', 'DE', 'ES', 'FR', 'IT', 'JA', 'KO', 'PT', 'RU', 'ZH'],
  'mlx-community/parakeet-tdt-0.6b-v3':              ['EN', 'BG', 'CS', 'DA', 'DE', 'EL', 'ES', 'ET', 'FI', 'HR', 'HU', 'IT', 'LT', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'RU', 'SK', 'SL', 'SV', 'UK'],
  'mlx-community/VibeVoice-ASR-bf16':                '*',
};

/** Language names valid for a given model. Always includes 'Detect'. */
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

/** Model values compatible with a given language name. 'Detect' returns '*'. */
export function getCompatibleModelValues(languageName: string): string[] | '*' {
  if (!languageName || languageName === 'Detect') return '*';
  const code = LANGUAGE_CODE[languageName];
  if (!code) return '*';
  return Object.entries(MODEL_SUPPORTED_LANGS)
    .filter(([, langs]) => langs === '*' || langs.includes(code))
    .map(([model]) => model);
}