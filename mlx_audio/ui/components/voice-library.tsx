"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Bookmark, ChevronDown, Play } from "lucide-react"

type Voice = {
  id: string
  name: string
  language: string
  gender: "Male" | "Female"
  age: string
  accent: string
  region: string
  isSelected?: boolean
  tags?: string[]
}

const voices: Voice[] = [
  {
    id: "trustworthy-man",
    name: "Trustworthy Man",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Resonate",
    region: "EN-US (General)",
    isSelected: true,
  },
  {
    id: "expressive-narrator",
    name: "Expressive Narrator",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Audiobook",
    region: "EN-British",
  },
  {
    id: "radiant-girl",
    name: "Radiant Girl",
    language: "English",
    gender: "Female",
    age: "Young Adult",
    accent: "Lively",
    region: "EN-US (General)",
  },
  {
    id: "magnetic-voiced-male",
    name: "Magnetic-voiced Male",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Ad",
    region: "EN-US (General)",
  },
  {
    id: "compelling-lady",
    name: "Compelling Lady",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Broadcast",
    region: "EN-British",
  },
  {
    id: "aussie-bloke",
    name: "Aussie Bloke",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Bright",
    region: "EN-Australian",
  },
  {
    id: "captivating-female",
    name: "Captivating Female",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "News Report",
    region: "EN-US (General)",
  },
  {
    id: "upbeat-woman",
    name: "Upbeat Woman",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Bright",
    region: "EN-US (General)",
  },
  {
    id: "calm-woman",
    name: "Calm Woman",
    language: "English",
    gender: "Female",
    age: "Adult",
    accent: "Audiobook",
    region: "EN-US (General)",
  },
  {
    id: "upset-girl",
    name: "Upset Girl",
    language: "English",
    gender: "Female",
    age: "Young Adult",
    accent: "Sad",
    region: "EN-British",
  },
  {
    id: "gentle-voiced-man",
    name: "Gentle-voiced Man",
    language: "English",
    gender: "Male",
    age: "Adult",
    accent: "Resonate",
    region: "EN-US (General)",
  },
]

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
    initialSelectedVoice
      ? voices.find((v) => v.name === initialSelectedVoice)?.id || "trustworthy-man"
      : "trustworthy-man",
  )
  const [language, setLanguage] = useState("")
  const [accent, setAccent] = useState("")
  const [gender, setGender] = useState("")
  const [age, setAge] = useState("")
  const [bookmarkedVoices, setBookmarkedVoices] = useState<string[]>([])
  const [isCloneModalOpen, setIsCloneModalOpen] = useState(false)

  useEffect(() => {
    if (initialSelectedVoice) {
      const voiceId = voices.find((v) => v.name === initialSelectedVoice)?.id
      if (voiceId) {
        setSelectedVoice(voiceId)
      }
    }
  }, [initialSelectedVoice])

  const getGradientForVoice = (voiceId: string) => {
    // Map of voice IDs to gradient classes
    const gradientMap: Record<string, string> = {
      "trustworthy-man": "bg-gradient-to-br from-blue-400 to-indigo-600",
      "expressive-narrator": "bg-gradient-to-br from-purple-400 to-indigo-500",
      "radiant-girl": "bg-gradient-to-br from-pink-400 to-orange-300",
      "magnetic-voiced-male": "bg-gradient-to-br from-sky-400 to-blue-600",
      "compelling-lady": "bg-gradient-to-br from-rose-400 to-red-500",
      "aussie-bloke": "bg-gradient-to-br from-amber-400 to-orange-500",
      "captivating-female": "bg-gradient-to-br from-teal-400 to-emerald-500",
      "upbeat-woman": "bg-gradient-to-br from-green-400 to-emerald-500",
      "calm-woman": "bg-gradient-to-br from-indigo-400 to-purple-500",
      "upset-girl": "bg-gradient-to-br from-rose-300 to-pink-500",
      "gentle-voiced-man": "bg-gradient-to-br from-cyan-400 to-blue-500",
    }

    // Return the gradient class or a default gradient if not found
    return gradientMap[voiceId] || "bg-gradient-to-br from-gray-400 to-gray-600"
  }

  const handleSelectVoice = (voiceId: string) => {
    setSelectedVoice(voiceId)
    // Get the voice name from the voices array
    const selectedVoiceName = voices.find((v) => v.id === voiceId)?.name || "Trustworthy Man"

    // Call the onSelectVoice callback if provided
    if (onSelectVoice) {
      onSelectVoice(selectedVoiceName)
    }

    // In a real app, this would update the selected voice in the parent component
    if (onClose) {
      setTimeout(() => {
        onClose()
      }, 300)
    }
  }

  const handleBookmark = (e: React.MouseEvent, voiceId: string) => {
    e.stopPropagation()
    setBookmarkedVoices((prev) => (prev.includes(voiceId) ? prev.filter((id) => id !== voiceId) : [...prev, voiceId]))
  }

  const handleUseVoice = (e: React.MouseEvent, voiceId: string) => {
    e.stopPropagation()
    // Set the selected voice
    setSelectedVoice(voiceId)

    // Get the voice name from the voices array
    const selectedVoiceName = voices.find((v) => v.id === voiceId)?.name || "Trustworthy Man"

    // Call the onSelectVoice callback if provided
    if (onSelectVoice) {
      onSelectVoice(selectedVoiceName)
    }

    // Provide visual feedback
    const voiceName = voices.find((v) => v.id === voiceId)?.name
    console.log(`Voice selected: ${voiceName}`)
  }

  const handleCreateVoice = () => {
    setIsCloneModalOpen(true)
  }

  return (
    <div
      className="flex flex-col h-full"
      onClick={() => console.log("Current selected voice:", selectedVoice)}
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
                  <button
                    className={`${bookmarkedVoices.includes(voice.id) ? "text-yellow-500" : "text-gray-400"} hover:text-yellow-500`}
                    onClick={(e) => handleBookmark(e, voice.id)}
                  >
                    <Bookmark className="h-4 w-4" />
                  </button>
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
