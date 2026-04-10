import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, Check } from 'lucide-react';

const LANGUAGES = [
  "Detect",
  "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani",
  "Basque", "Belarusian", "Bengali", "Bosnian", "Bulgarian",
  "Catalan", "Chinese", "Croatian", "Czech",
  "Danish", "Dutch",
  "English", "Estonian",
  "Finnish", "French",
  "Galician", "Georgian", "German", "Greek", "Gujarati",
  "Hebrew", "Hindi", "Hungarian",
  "Icelandic", "Indonesian", "Irish", "Italian",
  "Japanese", "Javanese",
  "Kannada", "Kazakh", "Korean",
  "Latvian", "Lithuanian",
  "Macedonian", "Malay", "Malayalam", "Maltese", "Marathi", "Mongolian",
  "Nepali", "Norwegian",
  "Persian", "Polish", "Portuguese", "Punjabi",
  "Romanian", "Russian",
  "Serbian", "Slovak", "Slovenian", "Somali", "Spanish", "Swahili", "Swedish",
  "Tamil", "Telugu", "Thai", "Turkish",
  "Ukrainian", "Urdu", "Uzbek",
  "Vietnamese",
  "Welsh",
  "Yoruba",
  "Zulu",
];

interface Props {
  value: string;
  onChange: (value: string) => void;
  allowedLanguages?: string[] | '*';
}

export function SearchableLanguageSelect({ value, onChange, allowedLanguages }: Props) {
  const [isOpen, setIsOpen]           = useState(false);
  const [search, setSearch]           = useState('');
  const [highlighted, setHighlighted] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef     = useRef<HTMLInputElement>(null);
  const listRef      = useRef<HTMLUListElement>(null);
  const filtered = LANGUAGES.filter(l =>
    l.toLowerCase().includes(search.toLowerCase())
  );
  const isCompatible = (lang: string) =>
    !allowedLanguages || allowedLanguages === '*' || allowedLanguages.includes(lang);
  
  // Close on outside click
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

  // Focus search input when opened; pre-scroll to selected item
  useEffect(() => {
    if (!isOpen) return;
    inputRef.current?.focus();
    const idx = filtered.indexOf(value);
    setHighlighted(idx >= 0 ? idx : 0);
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // Scroll highlighted item into view
  useEffect(() => {
    const item = listRef.current?.children[highlighted] as HTMLElement | undefined;
    item?.scrollIntoView({ block: 'nearest' });
  }, [highlighted]);

  const select = (lang: string) => {
    onChange(lang);
    setIsOpen(false);
    setSearch('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) {
      if (['Enter', ' ', 'ArrowDown'].includes(e.key)) {
        e.preventDefault();
        setIsOpen(true);
      }
      return;
    }
    if (e.key === 'ArrowDown') { e.preventDefault(); setHighlighted(i => Math.min(i + 1, filtered.length - 1)); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setHighlighted(i => Math.max(i - 1, 0)); }
    if (e.key === 'Enter')     { e.preventDefault(); filtered[highlighted] && select(filtered[highlighted]); }
    if (e.key === 'Escape')    { setIsOpen(false); setSearch(''); }
  };

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
        <span className={value ? '' : 'text-gray-400 dark:text-gray-500'}>
          {value || 'Select language'}
        </span>
        <ChevronDown
          className={`h-5 w-5 text-gray-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown panel */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-1 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg overflow-hidden">

          {/* Search field */}
          <div className="p-2 border-b border-gray-100 dark:border-gray-700">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400 pointer-events-none" />
              <input
                ref={inputRef}
                type="text"
                value={search}
                onChange={e => { setSearch(e.target.value); setHighlighted(0); }}
                placeholder="Type to filter…"
                className="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500"
              />
            </div>
          </div>

          {/* Options */}
          <ul ref={listRef} role="listbox" className="max-h-60 overflow-y-auto py-1">
            {filtered.length === 0 ? (
              <li className="px-4 py-2.5 text-sm text-gray-400 text-center">
                No languages found
              </li>
            ) : (
              filtered.map((lang, i) => {
                const compatible = isCompatible(lang);
                return (
                <li
                  key={lang}
                  role="option"
                  aria-selected={lang === value}
                  aria-disabled={!compatible}
                  onClick={() => compatible && select(lang)}
                  onMouseEnter={() => compatible && setHighlighted(i)}
                  className={[
                    'flex items-center justify-between px-4 py-2 text-sm cursor-pointer select-none',
                    compatible ? 'cursor-pointer' : 'cursor-not-allowed opacity-35',
                    compatible && i === highlighted
                      ? 'bg-sky-50 dark:bg-sky-900/30 text-sky-600 dark:text-sky-400'
                      : compatible ? 'hover:bg-gray-50 dark:hover:bg-gray-700/50' : '',
                  ].join(' ')}
                >
                  <span>{lang}</span>
                  {lang === value && compatible && (
                    <Check className="h-4 w-4 text-sky-500 shrink-0" />
                  )}
                </li>
                );
              })
            )}
          </ul>
        </div>
      )}
    </div>
  );
}