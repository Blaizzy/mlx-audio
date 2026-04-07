"use client"

import { useState, useEffect, type MouseEvent } from "react"
import { Play } from "lucide-react"

type Voice = {
  id: string
  name: string
  language: string
  gender: "Male" | "Female"
  age: string
  accent: string
  region: string
  tags?: string[]
}

const voices: Voice[] = [
  {
    id: "af_heart",
    name: "Heart",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Warm",
    region: "EN-US (American)",
  },
  {
    id: "af_bella",
    name: "Bella",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Bright",
    region: "EN-US (American)",
  },
  {
    id: "af_nova",
    name: "Nova",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Clear",
    region: "EN-US (American)",
  },
  {
    id: "af_sky",
    name: "Sky",
    language: "English",
    gender: "Female",
    age: "Young Adult",
    accent: "Lively",
    region: "EN-US (American)",
  },
  {
    id: "am_adam",
    name: "Adam",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Deep",
    region: "EN-US (American)",
  },
  {
    id: "am_echo",
    name: "Echo",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Resonant",
    region: "EN-US (American)",
  },
  {
    id: "bf_alice",
    name: "Alice",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Refined",
    region: "EN-British",
  },
  {
    id: "bf_emma",
    name: "Emma",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Clear",
    region: "EN-British",
  },
  {
    id: "bm_daniel",
    name: "Daniel",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Deep",
    region: "EN-British",
  },
  {
    id: "bm_george",
    name: "George",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Warm",
    region: "EN-British",
  },
]

export function getVoiceDisplayName(voiceId: string): string {
  const voice = voices.find(v => v.id === voiceId)
  return voice?.name || voiceId
}

export const VOICE_GRADIENT_COLORS: Record<string, string> = {
  af_heart: "from-pink-400 to-rose-500",
  af_bella: "from-purple-400 to-pink-500",
  af_nova: "from-sky-400 to-blue-500",
  af_sky: "from-cyan-400 to-sky-500",
  am_adam: "from-blue-400 to-indigo-600",
  am_echo: "from-indigo-400 to-purple-500",
  bf_alice: "from-rose-400 to-pink-500",
  bf_emma: "from-amber-400 to-orange-500",
  bm_daniel: "from-slate-400 to-gray-600",
  bm_george: "from-teal-400 to-emerald-500",
}

interface VoiceLibraryProps {
  onClose?: () => void
  onSelectVoice?: (voice: string) => void
  hideFreeTrial?: boolean
  initialSelectedVoice?: string
}

export function VoiceLibrary({
  onClose,
  onSelectVoice,
  hideFreeTrial = false,
  initialSelectedVoice,
}: VoiceLibraryProps) {
  const [activeTab, setActiveTab] = useState<"library" | "my-voices">("library")
  const [selectedVoice, setSelectedVoice] = useState(
    initialSelectedVoice || "af_heart",
  )
  const [language, setLanguage] = useState("")
  const [accent, setAccent] = useState("")
  const [gender, setGender] = useState("")
  const [age, setAge] = useState("")
  const [isCloneModalOpen, setIsCloneModalOpen] = useState(false)

  useEffect(() => {
    if (initialSelectedVoice) {
      setSelectedVoice(initialSelectedVoice)
    }
  }, [initialSelectedVoice])

  const getGradientForVoice = (voiceId: string) =>
    `bg-gradient-to-br ${VOICE_GRADIENT_COLORS[voiceId] || "from-gray-400 to-gray-600"}`

  const handleSelectVoice = (voiceId: string) => {
    setSelectedVoice(voiceId)
    if (onSelectVoice) {
      onSelectVoice(voiceId)
    }
    if (onClose) {
      setTimeout(() => {
        onClose()
      }, 300)
    }
  }

  const handleUseVoice = (e: MouseEvent, voiceId: string) => {
    e.stopPropagation()
    setSelectedVoice(voiceId)
    if (onSelectVoice) {
      onSelectVoice(voiceId)
    }
  }

  const handleCreateVoice = () => {
    setIsCloneModalOpen(true)
  }

  return (
    <div
      className="flex flex-col h-full"
      style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100%" }}
    >
      <div className="overflow-y-auto">
        <div className="space-y-2">
          {activeTab === "library" ? (
            voices.map((voice) => (
              <div
                key={voice.id}
                className="flex items-center justify-between border border-gray-200 dark:border-gray-700 rounded-md p-2 hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
                onClick={() => handleSelectVoice(voice.id)}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-10 h-10 rounded-md flex-shrink-0 ${getGradientForVoice(voice.id)}`}
                    aria-label={`${voice.name} avatar`}
                  ></div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium">{voice.name}</span>
                    </div>
                    <div className="flex flex-wrap gap-1 text-xs text-gray-500 dark:text-gray-400">
                      <span>{voice.language}</span>
                      <span>•</span>
                      <span>{voice.gender === "Male" ? "Male" : "Female"}</span>
                      <span>•</span>
                      <span>{voice.age}</span>
                      {voice.accent && (
                        <>
                          <span>•</span>
                          <span>{voice.accent}</span>
                        </>
                      )}
                      <span>•</span>
                      <span>{voice.region}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {voice.id === selectedVoice ? (
                    <span className="bg-sky-500 text-white text-xs px-2 py-1 rounded-md">Selected</span>
                  ) : (
                    <button
                      className="bg-black dark:bg-white text-white dark:text-black text-xs px-2 py-1 rounded-md flex items-center"
                      onClick={(e) => handleUseVoice(e, voice.id)}
                    >
                      <Play className="h-3 w-3 mr-1" />
                      Use
                    </button>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="py-8 text-center text-sm text-gray-500 dark:text-gray-400">
              <p>You haven't created any custom voices yet.</p>
              <button
                className="mt-4 rounded-md bg-sky-500 dark:bg-sky-600 px-4 py-2 text-sm text-white hover:bg-sky-600 dark:hover:bg-sky-700"
                onClick={handleCreateVoice}
              >
                Create Your First Voice
              </button>
            </div>
          )}
        </div>
      </div>


    </div>
  )
}
