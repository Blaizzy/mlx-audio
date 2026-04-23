import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, Check, ExternalLink } from 'lucide-react';

interface ModelOption {
  value: string;
  label: string;
  description?: string;
}

interface ModelGroup {
  group: string;
  models: ModelOption[];
}
const MODEL_GROUPS: ModelGroup[] = [
  {
    group: 'Whisper',
    models: [
      { value: 'mlx-community/whisper-large-v3-turbo-asr-fp16',    label: 'Whisper Large v3 Turbo ASR fp16', description: 'ASR-optimised, fp16 (99+ languages)' },
      { value: 'distil-whisper/distil-large-v3',                    label: 'Distil Large v3',                 description: 'Distilled, ~756M params (EN)' },
    ],
  },
  { 
    group: 'Moonshine',
    models: [
      { value: 'UsefulSensors/moonshine-tiny',                    label: 'Moonshine Tiny',                  description: 'Smallest variant (EN)'},
      { value: 'UsefulSensors/moonshine-base',                    label: 'Moonshine Base',                  description: 'Larger, more accurate (EN)'},
    ],
  },
  { 
    group: 'MMS ASR',
    models: [
      { value: 'facebook/mms-1b-fl102',                    label: 'MMS FL102',                  description: 'Finetuned on FLEURS (102 languages)'},
      { value: 'facebook/mms-1b-all',                      label: 'MMS All',                    description: 'All supported languages (1162 languages)'},
    ],
  }, 
  { 
    group: 'Granite',
    models: [
      { value: 'fibm-granite/granite-4.0-1b-speech',       label: 'Granite',                  description: 'Speech recognition and translation (EN, DE, ES, JA, PT)'},
    ],
  },   
  {
    group: 'Voxtral',
    models: [
      { value: 'mlx-community/Voxtral-Mini-3B-2507-bf16',          label: 'Voxtral Mini 3B bf16',            description: 'Full precision (EN, DE, ES, HI, JA, PT), Autodetect of language less reliable'},
      { value: 'mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit', label: 'Voxtral Mini 4B Realtime 4bit',   description: 'Streaming, Quantized, faster (EN, AR, DE, ES, FR, HI, IT, JA, KO, NL, PT, RU, ZH)' },
      { value: 'mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16', label: 'Voxtral Mini 4B Realtime fp16',   description: 'Streaming, Full precision (EN, AR, DE, ES, FR, HI, IT, JA, KO, NL, PT, RU, ZH)' },
    ],
  },
  {
    group: 'Qwen',
    models: [
      { value: 'mlx-community/Qwen2-Audio-7B-Instruct-4bit',       label: 'Qwen2 Audio 7B Instruct 4bit',   description: 'ASR, captioning, emotion, translation (EN, FR, YUE, FR)' },
      { value: 'mlx-community/Qwen3-ASR-1.7B-8bit',                label: 'Qwen3 ASR 1.7B 8bit',            description: '1.7B params, quantized (EN, AR, CS, DA, DE, EL, ES, FA, FIL, FR, HI, HU, ID, IT, JA, KO, MK, MS, NL, PT, RO, RU, SV, TH, VI, YUE, ZH) ' },
      { value: 'mlx-community/Qwen3-ForcedAligner-0.6B-8bit',      label: 'Qwen3 ForcedAligner 0.6B 8bit',  description: 'Forced alignment, 0.6B (EN, DE, ES, FR, IT, JA, KO, PT, RU, YUE, ZH)' },
    ],
  },
  {
    group: 'Parakeet',
    models: [
      { value: 'mlx-community/parakeet-tdt-0.6b-v3',               label: 'Parakeet TDT 0.6B v3',           description: 'NVIDIA, 0.6B params (EN, BG, CS, DA, DE, EL, ES, ET, FI, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK)' },
    ],
  },
  {
    group: 'VibeVoice',
    models: [
      { value: 'mlx-community/VibeVoice-ASR-bf16',                  label: 'VibeVoice ASR bf16',              description: 'Full precision (50+ languages)' },
    ],
  },
];

// Flat list used for keyboard navigation across groups
const ALL_MODELS: ModelOption[] = MODEL_GROUPS.flatMap(g => g.models);

interface Props {
  value: string;
  onChange: (value: string) => void;
  allowedValues?: string[] | '*';  
}

export function SearchableSTTModelSelect({ value, onChange, allowedValues }: Props) {
  const [isOpen, setIsOpen]           = useState(false);
  const [search, setSearch]           = useState('');
  const [highlighted, setHighlighted] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef     = useRef<HTMLInputElement>(null);
  const listRef      = useRef<HTMLDivElement>(null);

  // Filtered flat list
  const filteredModels = ALL_MODELS.filter(m =>
    m.value.toLowerCase().includes(search.toLowerCase()) ||
    m.label.toLowerCase().includes(search.toLowerCase())
  );

  // When search has a value that matches nothing, offer it as custom
  const isCustom    = search.length > 0 && !ALL_MODELS.some(m => m.value === search);
  const showCustom  = isCustom && search.includes('/');
  const isCompatible = (modelValue: string) =>
    !allowedValues || allowedValues === '*' || allowedValues.includes(modelValue);
  const compatibleCount = filteredModels.filter(m => isCompatible(m.value)).length;
  const totalItems      = compatibleCount + (showCustom ? 1 : 0);

  // Resolve displayed label for current value
  const selectedLabel = ALL_MODELS.find(m => m.value === value)?.value ?? value;

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) {
        setIsOpen(false);
        setSearch('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    inputRef.current?.focus();
    const idx = filteredModels.findIndex(m => m.value === value);
    setHighlighted(idx >= 0 ? idx : 0);
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const item = listRef.current?.querySelectorAll('[data-option]')[highlighted] as HTMLElement | undefined;
    item?.scrollIntoView({ block: 'nearest' });
  }, [highlighted]);

  const select = (val: string) => {
    onChange(val);
    setIsOpen(false);
    setSearch('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) {
      if (['Enter', ' ', 'ArrowDown'].includes(e.key)) { e.preventDefault(); setIsOpen(true); }
      return;
    }
    if (e.key === 'ArrowDown') { e.preventDefault(); setHighlighted(i => Math.min(i + 1, totalItems - 1)); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setHighlighted(i => Math.max(i - 1, 0)); }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (highlighted < filteredModels.length) {
        filteredModels[highlighted] && select(filteredModels[highlighted].value);
      } else if (showCustom) {
        select(search);
      }
    }
    if (e.key === 'Escape') { setIsOpen(false); setSearch(''); }
  };

  // Build grouped filtered view
  const filteredGroups = MODEL_GROUPS.map(g => ({
    ...g,
    models: g.models.filter(m => filteredModels.includes(m)),
  })).filter(g => g.models.length > 0);

  // Running index for highlight tracking across groups
  let runningIndex = 0;

  return (
    <div ref={containerRef} className="relative" onKeyDown={handleKeyDown}>

      {/* Trigger */}
      <button
        type="button"
        onClick={() => setIsOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        className="w-full flex items-center justify-between rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2.5 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500 text-left"
      >
        <span className={`truncate font-mono text-sm ${value ? '' : 'text-gray-400 dark:text-gray-500'}`}>
          {selectedLabel || 'Select model'}
        </span>
        <ChevronDown
          className={`ml-2 h-5 w-5 text-gray-400 shrink-0 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-1 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg overflow-hidden">

          {/* Search */}
          <div className="p-2 border-b border-gray-100 dark:border-gray-700">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400 pointer-events-none" />
              <input
                ref={inputRef}
                type="text"
                value={search}
                onChange={e => { setSearch(e.target.value); setHighlighted(0); }}
                placeholder="Filter or paste a model path…"
                className="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500 font-mono"
              />
            </div>
          </div>

          {/* Options */}
          <div ref={listRef} role="listbox" className="max-h-72 overflow-y-auto py-1">

            {/* Custom model entry */}
            {showCustom && (
              <div
                data-option
                role="option"
                onClick={() => select(search)}
                onMouseEnter={() => setHighlighted(filteredModels.length)}
                className={[
                  'flex items-center gap-2 px-4 py-2 text-sm cursor-pointer border-b border-gray-100 dark:border-gray-700',
                  highlighted === filteredModels.length
                    ? 'bg-sky-50 dark:bg-sky-900/30 text-sky-600 dark:text-sky-400'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-700/50 text-gray-500 dark:text-gray-400',
                ].join(' ')}
              >
                <ExternalLink className="h-3.5 w-3.5 shrink-0" />
                <span className="font-mono truncate">Use: {search}</span>
              </div>
            )}

            {filteredGroups.length === 0 && !showCustom && (
              <div className="px-4 py-2.5 text-sm text-gray-400 text-center">
                No models found — paste a full HuggingFace path
              </div>
            )}

            {filteredGroups.map(group => (
              <div key={group.group}>
                {/* Group header */}
                <div className="px-3 pt-2 pb-1 text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider">
                  {group.group}
                </div>

                {group.models.map(model => {
                  const idx = runningIndex++;
                  const compatible = isCompatible(model.value);
                  return (
                    <div
                      key={model.value}
                      data-option={compatible ? true : undefined}
                      role="option"
                      aria-selected={model.value === value}
                      aria-disabled={!compatible}
                      onClick={() => compatible && select(model.value)}
                      onMouseEnter={() => compatible && setHighlighted(idx)}
                      className={[
                        'flex items-center justify-between px-4 py-2 select-none',
                        compatible ? 'cursor-pointer' : 'cursor-not-allowed opacity-35',
                        compatible && idx === highlighted
                          ? 'bg-sky-50 dark:bg-sky-900/30'
                          : compatible ? 'hover:bg-gray-50 dark:hover:bg-gray-700/50' : '',
                      ].join(' ')}
                    >
                      <div className="min-w-0">
                        <div className={`text-sm font-mono truncate ${compatible && idx === highlighted ? 'text-sky-600 dark:text-sky-400' : ''}`}>
                          {model.value}
                        </div>
                        {model.description && (
                          <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                            {model.description}
                          </div>
                        )}
                      </div>
                      {model.value === value && compatible && (
                        <Check className="h-4 w-4 text-sky-500 shrink-0 ml-2" />
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}