"use client"

import type React from "react"

import { useState, useRef, useMemo, useEffect } from "react"
import { ChevronDown, Download, ThumbsUp, ThumbsDown, Play, Pause, RefreshCw } from "lucide-react"
import { LayoutWrapper } from "@/components/layout-wrapper"

// Custom range input component with colored progress
function RangeInput({
  min,
  max,
  step,
  value,
  onChange,
  className = "",
  ariaLabel,
}: {
  min: number
  max: number
  step: number
  value: number
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  className?: string
  ariaLabel?: string
}) {
  const percentage = ((value - Number(min)) / (Number(max) - Number(min))) * 100

  return (
    <div className={`relative flex-1 ${className}`}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        className="w-full appearance-none bg-transparent cursor-pointer"
        aria-label={ariaLabel}
        style={{
          background: `linear-gradient(to right, #0ea5e9 0%, #0ea5e9 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`,
          height: "2px",
          borderRadius: "2px",
        }}
      />
    </div>
  )
}

const LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto detect (match text)", defaultVoice: "af_heart" },
  { value: "a", label: "English (American)", defaultVoice: "af_heart" },
  { value: "b", label: "English (British)", defaultVoice: "bf_emma" },
  { value: "e", label: "Spanish", defaultVoice: "ef_dora" },
  { value: "f", label: "French", defaultVoice: "ff_siwis" },
  { value: "h", label: "Hindi", defaultVoice: "hf_alpha" },
  { value: "i", label: "Italian", defaultVoice: "if_sara" },
  { value: "p", label: "Portuguese (Brazil)", defaultVoice: "pf_dora" },
  { value: "j", label: "Japanese", defaultVoice: "jf_alpha" },
  { value: "z", label: "Mandarin Chinese", defaultVoice: "zf_xiaobei" },
]

const MODEL_LANGUAGE_SUPPORT: Record<string, string[]> = {
  marvis: ["a"],
  kokoro: LANGUAGE_OPTIONS.filter((option) => option.value !== "auto").map(
    (option) => option.value
  ),
  spark: ["a", "b"],
  default: ["a"],
}

const getModelFamily = (modelId: string) => {
  const normalized = modelId.toLowerCase()
  if (normalized.includes("marvis")) return "marvis"
  if (normalized.includes("kokoro")) return "kokoro"
  if (normalized.includes("spark")) return "spark"
  return "default"
}

const isMarvisModel = (modelId: string) => modelId.toLowerCase().includes("marvis")

const getDefaultVoice = (modelId: string, langCode: string) => {
  if (isMarvisModel(modelId)) {
    return "conversational_a"
  }
  const option = LANGUAGE_OPTIONS.find((entry) => entry.value === langCode)
  return option?.defaultVoice ?? "af_heart"
}

const detectLanguageFromText = (text: string): string | null => {
  const content = text.trim()
  if (!content) {
    return null
  }

  const hasHiragana = /[\u3040-\u309F]/.test(content)
  const hasKatakana = /[\u30A0-\u30FF]/.test(content)
  const hasCJK = /[\u4E00-\u9FFF]/.test(content)
  if (hasHiragana || hasKatakana) {
    return "j"
  }
  if (hasCJK) {
    return "z"
  }

  const accentedSpanish = /[áéíóúñüÁÉÍÓÚÑÜ]/.test(content)
  if (accentedSpanish) {
    return "e"
  }

  const accentedFrench = /[àâçéèêëîïôûùüÿÀÂÇÉÈÊËÎÏÔÛÙÜŸ]/.test(content)
  if (accentedFrench) {
    return "f"
  }

  return null
}

export default function SpeechSynthesis() {
  const [text, setText] = useState("But I also have other interests, such as playing tic-tac-toe.")
  const [isPlaying, setIsPlaying] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [pitch, setPitch] = useState(0)
  const [volume, setVolume] = useState(1)
  const [currentTime, setCurrentTime] = useState("00:00")
  const [duration, setDuration] = useState("00:04")
  const [activeTab, setActiveTab] = useState<"settings" | "history">("settings")
  const [model, setModel] = useState("Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit")
  const [language, setLanguage] = useState("auto")
  const [liked, setLiked] = useState<boolean | null>(null)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>(null)
  const modelFamily = getModelFamily(model)
  const allowedLanguages = useMemo(
    () => MODEL_LANGUAGE_SUPPORT[modelFamily] ?? MODEL_LANGUAGE_SUPPORT.default,
    [modelFamily]
  )
  const autoAllowed = allowedLanguages.length > 1

  useEffect(() => {
    if (!autoAllowed && language === "auto") {
      setLanguage(allowedLanguages[0])
    } else if (
      language !== "auto" &&
      !allowedLanguages.includes(language)
    ) {
      setLanguage(autoAllowed ? "auto" : allowedLanguages[0])
    }
  }, [autoAllowed, allowedLanguages, language])

  const languageOptionsForModel = useMemo(
    () =>
      LANGUAGE_OPTIONS.filter((option) =>
        option.value === "auto"
          ? autoAllowed
          : allowedLanguages.includes(option.value)
      ),
    [allowedLanguages, autoAllowed]
  )

  const languageLabel =
    language === "auto"
      ? "Auto detect (match text)"
      : LANGUAGE_OPTIONS.find((option) => option.value === language)?.label ??
        "Auto detect (match text)"

  const detectedLanguage =
    language === "auto" && autoAllowed ? detectLanguageFromText(text) : null
  const resolvedLanguage =
    language === "auto"
      ? detectedLanguage ?? allowedLanguages[0]
      : language
  const selectedVoice = getDefaultVoice(model, resolvedLanguage)

  const cleanupAudioUrl = () => {
    if (audioRef.current?.src && audioRef.current.src.startsWith("blob:")) {
      URL.revokeObjectURL(audioRef.current.src)
    }
    setCurrentAudioUrl(null)
  }

  const handlePlaybackError = (error: unknown) => {
    if (error instanceof DOMException && error.name === "AbortError") {
      return
    }
    console.error("Audio playback failed:", error)
  }

  const playAudio = async () => {
    if (!audioRef.current) return
    try {
      await audioRef.current.play()
      setIsPlaying(true)
    } catch (error) {
      handlePlaybackError(error)
    }
  }

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
  }

  const handlePlayPause = () => {
    if (!audioRef.current || !audioRef.current.src || audioRef.current.src === window.location.href) {
      // If no audio is loaded, or src is not a valid audio source, try to generate first.
      // This can happen if the user clicks play before generating or after an error.
      handleGenerate()
      return
    }

    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      void playAudio()
    }
  }

  const handleGenerate = async () => {
    if (!audioRef.current) return
    setIsGenerating(true)
    audioRef.current.pause()
    setIsPlaying(false)
    cleanupAudioUrl()

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost';
    const API_PORT = process.env.NEXT_PUBLIC_API_PORT || '8000';

    const voice = selectedVoice.trim() || getDefaultVoice(model, resolvedLanguage)

    try {
      const response = await fetch(`${API_BASE_URL}:${API_PORT}/v1/audio/speech`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: model, // Or the specific model identifier if different
          input: text,
          voice: voice,
          speed: speed,
          ...(resolvedLanguage ? { lang_code: resolvedLanguage } : {}),
          // pitch and other parameters can be added here if supported by the backend
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const blob = await response.blob()
      const audioUrl = URL.createObjectURL(blob)
      audioRef.current.src = audioUrl
      setCurrentAudioUrl(audioUrl)

      audioRef.current.onloadedmetadata = () => {
        setDuration(formatTime(Math.floor(audioRef.current?.duration || 0)))
        setCurrentTime("00:00")
        void playAudio()
      }

      audioRef.current.ontimeupdate = () => {
        setCurrentTime(formatTime(Math.floor(audioRef.current?.currentTime || 0)))
      }

      audioRef.current.onended = () => {
        setIsPlaying(false)
        setCurrentTime("00:00")
         // Revoke the object URL to free up resources
        cleanupAudioUrl()
      }
    } catch (error) {
      console.error("Error generating speech:", error)
      // Handle error appropriately in the UI
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = async () => {
    const src = audioRef.current?.src
    if (!src || src === window.location.href) {
      alert("Generate audio before downloading.")
      return
    }

    let downloadUrl = src
    let revokeAfter = false

    if (!src.startsWith("blob:")) {
      try {
        const response = await fetch(src)
        const blob = await response.blob()
        downloadUrl = URL.createObjectURL(blob)
        revokeAfter = true
      } catch (error) {
        console.error("Failed to download audio:", error)
        alert("Unable to download audio. Please try regenerating.")
        return
      }
    }

    const link = document.createElement("a")
    link.href = downloadUrl
    link.download = "mlx-audio.wav"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    if (revokeAfter) {
      URL.revokeObjectURL(downloadUrl)
    }
  }

  const handleFeedback = (isPositive: boolean) => {
    setLiked(isPositive)
  }

  const getCharacterCount = () => {
    return text.length
  }

  const formatTime = (seconds: number) => {
    return `00:${seconds.toString().padStart(2, "0")}`
  }

  return (
    <LayoutWrapper activeTab="audio" activePage="text-to-speech">
      <div className="flex flex-1 overflow-hidden">
        {/* Text Input Area */}
        <div className="flex-1 overflow-auto border-r border-gray-200 dark:border-gray-700 p-6">
          <h1 className="mb-6 text-2xl font-bold">Speech Synthesis</h1>
          <textarea
            className="min-h-[200px] w-full resize-none rounded-md border border-gray-200 dark:border-gray-700 p-4 text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 focus:border-blue-500 focus:outline-none"
            value={text}
            onChange={handleTextChange}
            placeholder="Enter text to convert to speech..."
          />
          <div className="mt-auto flex items-center justify-between pt-4 text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-2">
              <span>Long Text</span>
              <div className="h-2 w-2 rounded-full bg-gray-400 dark:bg-gray-500"></div>
            </div>
            <div className="flex items-center space-x-2">
              <span>{getCharacterCount()} / 5,000 characters</span>
              <div className="h-2 w-2 rounded-full bg-gray-200 dark:bg-gray-600"></div>
            </div>
          </div>
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <select
                  className="flex w-44 appearance-none items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs pr-6 bg-white dark:bg-gray-800"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  {languageOptionsForModel.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-3 h-3 w-3 pointer-events-none" />
              </div>
              <p className="text-[10px] text-gray-500 dark:text-gray-400">
                {autoAllowed
                  ? "Auto-detect is available for this model."
                  : `This model supports: ${allowedLanguages
                      .map(
                        (code) =>
                          LANGUAGE_OPTIONS.find((opt) => opt.value === code)?.label ??
                          code
                      )
                      .join(", ")}`}
              </p>
              <button
                className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                onClick={handleDownload}
              >
                <Download className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              </button>
            </div>
            <div className="flex items-center space-x-2">
              <button
                className={`rounded-md bg-sky-500 dark:bg-sky-600 px-3 py-1 text-sm text-white flex items-center hover:bg-sky-600 dark:hover:bg-sky-700 ${isGenerating ? "animate-pulse" : ""}`}
                onClick={handleGenerate}
                disabled={isGenerating}
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Generate
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        <div className="w-80 overflow-auto p-4 bg-white dark:bg-gray-900">
          <div className="mb-4 flex space-x-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            <button
              className={`pb-2 text-sm ${activeTab === "settings" ? "border-b-2 border-black dark:border-white font-medium" : "text-gray-500 dark:text-gray-400"}`}
              onClick={() => setActiveTab("settings")}
            >
              Settings
            </button>
            <button
              className={`pb-2 text-sm ${activeTab === "history" ? "border-b-2 border-black dark:border-white font-medium" : "text-gray-500 dark:text-gray-400"}`}
              onClick={() => setActiveTab("history")}
            >
              History
            </button>
          </div>

          {activeTab === "settings" ? (
            <>
              <div className="mb-6">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Model</span>
                  <div className="relative">
                    <select
                      className="flex w-40 appearance-none items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-sm pr-8 bg-white dark:bg-gray-800"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                    >
                      <option value="Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit">Marvis-TTS-100m-v0.2</option>
                      <option value="Marvis-AI/marvis-tts-250m-v0.2-MLX-6bit">Marvis-TTS-250m-v0.2</option>
                      <option value="Marvis-AI/marvis-tts-250m-v0.1-MLX-6bit">Marvis-TTS-250m-v0.1</option>
                      <option value="mlx-community/Kokoro-82M-bf16">Kokoro</option>
                      <option value="mlx-community/Spark-TTS-0.5B-bf16">SparkTTS</option>
                    </select>
                    <ChevronDown className="absolute right-2 top-2 h-4 w-4 pointer-events-none" />
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Speed</span>
                  <div className="flex items-center">
                    <div className="flex space-x-2 mr-2">
                      <button
                        onClick={() => setSpeed(0.5)}
                        className={`px-2 py-0.5 text-xs rounded-md ${speed === 0.5 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        0.5x
                      </button>
                      <button
                        onClick={() => setSpeed(1)}
                        className={`px-2 py-0.5 text-xs rounded-md ${speed === 1 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        1x
                      </button>
                      <button
                        onClick={() => setSpeed(1.5)}
                        className={`px-2 py-0.5 text-xs rounded-md ${speed === 1.5 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        1.5x
                      </button>
                    </div>
                    <span className="text-sm font-medium">{speed}x</span>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-2">Slow</span>
                  <RangeInput
                    min={0.5}
                    max={2}
                    step={0.1}
                    value={speed}
                    onChange={(e) => setSpeed(Number.parseFloat(e.target.value))}
                    ariaLabel="Speed control"
                  />
                  <span className="text-xs text-gray-500 ml-2">Fast</span>
                </div>
              </div>

              <div className="mb-6">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Pitch</span>
                  <div className="flex items-center">
                    <div className="flex space-x-2 mr-2">
                      <button
                        onClick={() => setPitch(-5)}
                        className={`px-2 py-0.5 text-xs rounded-md ${pitch === -5 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        Low
                      </button>
                      <button
                        onClick={() => setPitch(0)}
                        className={`px-2 py-0.5 text-xs rounded-md ${pitch === 0 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        Normal
                      </button>
                      <button
                        onClick={() => setPitch(5)}
                        className={`px-2 py-0.5 text-xs rounded-md ${pitch === 5 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        High
                      </button>
                    </div>
                    <span className="text-sm font-medium">{pitch}</span>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-2">-10</span>
                  <RangeInput
                    min={-10}
                    max={10}
                    step={1}
                    value={pitch}
                    onChange={(e) => setPitch(Number.parseInt(e.target.value))}
                    ariaLabel="Pitch control"
                  />
                  <span className="text-xs text-gray-500 ml-2">+10</span>
                </div>
              </div>

              <div className="mb-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm">Volume</span>
                  <div className="flex items-center">
                    <div className="flex space-x-2 mr-2">
                      <button
                        onClick={() => setVolume(0.5)}
                        className={`px-2 py-0.5 text-xs rounded-md ${volume === 0.5 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        Quiet
                      </button>
                      <button
                        onClick={() => setVolume(1)}
                        className={`px-2 py-0.5 text-xs rounded-md ${volume === 1 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        Normal
                      </button>
                      <button
                        onClick={() => setVolume(1.5)}
                        className={`px-2 py-0.5 text-xs rounded-md ${volume === 1.5 ? "bg-sky-100 dark:bg-sky-900 text-sky-600 dark:text-sky-300" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"}`}
                      >
                        Loud
                      </button>
                    </div>
                    <span className="text-sm font-medium">{volume}x</span>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-2">0</span>
                  <RangeInput
                    min={0}
                    max={2}
                    step={0.1}
                    value={volume}
                    onChange={(e) => setVolume(Number.parseFloat(e.target.value))}
                    ariaLabel="Volume control"
                  />
                  <span className="text-xs text-gray-500 ml-2">2</span>
                </div>
              </div>
            </>
          ) : (
            <div className="py-4 text-center text-sm text-gray-500 dark:text-gray-400">
              <p>No history available</p>
            </div>
          )}
        </div>
      </div>

      {/* Audio Player */}
      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 w-full">
        <div className="flex items-center w-full px-0">
          <button
            className="flex h-14 w-14 items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 ml-4"
            onClick={handlePlayPause}
          >
            {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
          </button>

          <div className="flex flex-col justify-between h-full flex-1 px-4 py-2">
            <div className="flex items-center justify-between w-full">
              <div className="text-sm">
                {languageLabel}: {text.length > 20 ? text.substring(0, 20) + "..." : text}
              </div>
              <div className="flex items-center space-x-2">
                <div className="text-xs text-gray-500 dark:text-gray-400 mr-2">How did this sound?</div>
                <button
                  className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={() => handleFeedback(true)}
                >
                  <ThumbsUp
                    className={`h-4 w-4 ${liked === true ? "text-sky-500" : "text-gray-500 dark:text-gray-400"}`}
                  />
                </button>
                <button
                  className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={() => handleFeedback(false)}
                >
                  <ThumbsDown
                    className={`h-4 w-4 ${liked === false ? "text-sky-500" : "text-gray-500 dark:text-gray-400"}`}
                  />
                </button>
                <button
                  className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={handleDownload}
                >
                  <Download className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>

            <div className="flex items-center mt-2">
              <div
                className="flex-1 bg-gray-200 dark:bg-gray-700 h-1 rounded-full cursor-pointer relative"
                onClick={(e) => {
                  if (!audioRef.current || !audioRef.current.duration) return
                  const bar = e.currentTarget
                  const rect = bar.getBoundingClientRect()
                  const position = (e.clientX - rect.left) / rect.width
                  audioRef.current.currentTime = position * audioRef.current.duration
                  setCurrentTime(formatTime(Math.floor(audioRef.current.currentTime)))
                }}
              >
                <div
                  className="bg-black dark:bg-white h-1 rounded-full absolute top-0 left-0"
                  style={{
                    width: audioRef.current?.duration
                      ? `${(audioRef.current.currentTime / audioRef.current.duration) * 100}%`
                      : "0%",
                  }}
                ></div>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 ml-4 whitespace-nowrap mr-4">
                {currentTime} / {duration}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hidden audio element for actual implementation */}
      <audio ref={audioRef} className="hidden" />
    </LayoutWrapper>
  )
}
