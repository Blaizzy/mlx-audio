"use client"

import { useState } from "react"
import { BookOpen, ShoppingBag, Wrench, ChevronDown, Download, ArrowRight, Play } from "lucide-react"
import { LayoutWrapper } from "@/components/layout-wrapper"
import Link from "next/link"
import { VoiceLibrary } from "@/components/voice-library"

export default function HomePage() {
  const [inputText, setInputText] = useState("")
  const [selectedModel, setSelectedModel] = useState("Kokoro")
  const [selectedVoice, setSelectedVoice] = useState("Trustworthy Man")
  const [activeButton, setActiveButton] = useState<string | null>(null)
  const [isVoiceModalOpen, setIsVoiceModalOpen] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const handleGenerate = async () => {
    const payload = {
      model: selectedModel,
      input: inputText,
      voice: selectedVoice,
    }

    try {
      const res = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })

      if (!res.ok) throw new Error("Request failed")
      const blob = await res.blob()
      setAudioUrl(URL.createObjectURL(blob))
    } catch (err) {
      console.error("Failed to generate speech", err)
    }
  }

  const getGradientForVoice = (name: string) => {
    if (name.includes("Man") || name.includes("Male")) {
      return "from-blue-400 to-indigo-600"
    } else if (name.includes("Girl") || name.includes("Female")) {
      return "from-pink-400 to-orange-300"
    } else if (name.includes("Narrator")) {
      return "from-purple-400 to-indigo-500"
    } else if (name.includes("Compelling")) {
      return "from-rose-400 to-red-500"
    } else if (name.includes("Magnetic")) {
      return "from-sky-400 to-blue-600"
    } else {
      return "from-gray-400 to-gray-600"
    }
  }

  return (
    <LayoutWrapper activeTab="audio" activePage="home">
      <main className="flex-1 overflow-auto p-6">
        <h1 className="text-2xl font-bold mb-4">Create Lifelike Speech</h1>

        {/* Text Input Area */}
        <div className="mb-6">
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
            <textarea
              className="w-full p-4 min-h-[120px] resize-none bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 focus:outline-none"
              placeholder="Start typing here to create lifelike speech in multiple languages, voices and emotions with MLX-Audio AI."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            ></textarea>

            {/* <div className="border-t border-gray-200 dark:border-gray-700 p-3 flex justify-between items-center">
              <div className="flex space-x-4">
                <button
                  onClick={() => setActiveButton(activeButton === "story" ? null : "story")}
                  className={`flex items-center space-x-2 text-sm px-3 py-1.5 rounded-md transition-colors ${
                    activeButton === "story"
                      ? "bg-sky-100 dark:bg-sky-900/30 text-sky-600 dark:text-sky-400"
                      : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`}
                >
                  <BookOpen className="h-4 w-4" />
                  <span>Tell a Story</span>
                </button>
                <button
                  onClick={() => setActiveButton(activeButton === "commercial" ? null : "commercial")}
                  className={`flex items-center space-x-2 text-sm px-3 py-1.5 rounded-md transition-colors ${
                    activeButton === "commercial"
                      ? "bg-sky-100 dark:bg-sky-900/30 text-sky-600 dark:text-sky-400"
                      : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`}
                >
                  <ShoppingBag className="h-4 w-4" />
                  <span>Create a Commercial</span>
                </button>
                <button
                  onClick={() => setActiveButton(activeButton === "tutor" ? null : "tutor")}
                  className={`flex items-center space-x-2 text-sm px-3 py-1.5 rounded-md transition-colors ${
                    activeButton === "tutor"
                      ? "bg-sky-100 dark:bg-sky-900/30 text-sky-600 dark:text-sky-400"
                      : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`}
                >
                  <Wrench className="h-4 w-4" />
                  <span>Build an AI Tutor</span>
                </button>
              </div>

              <div className="flex items-center space-x-2">
                <Download className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              </div>
            </div> */}

            <div className="border-t border-gray-200 dark:border-gray-700 p-3 flex justify-between items-center">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => alert(`Select model: ${selectedModel}`)}
                  className="flex items-center space-x-2 border border-gray-200 dark:border-gray-700 rounded-md px-2 py-1 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors cursor-pointer"
                >
                  <span className="text-sm text-gray-600 dark:text-gray-300">{selectedModel}</span>
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                </button>

                <button
                  onClick={() => setIsVoiceModalOpen(true)}
                  className="flex items-center space-x-2 border border-gray-200 dark:border-gray-700 rounded-md px-2 py-1 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors cursor-pointer"
                >
                  <div className={`h-5 w-5 bg-gradient-to-br ${getGradientForVoice(selectedVoice)} rounded-full`}></div>
                  <span className="text-sm text-gray-600 dark:text-gray-300">{selectedVoice}</span>
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                </button>
              </div>

              <button
                onClick={handleGenerate}
                className="bg-sky-500 hover:bg-sky-600 text-white px-4 py-1 rounded-md flex items-center space-x-1"
              >
                <span>Generate</span>
              </button>
            </div>
          </div>
        </div>

        {audioUrl && (
          <audio controls src={audioUrl} className="mt-4 w-full" />
        )}

      </main>
      {/* Voice Selection Modal */}
      {isVoiceModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 sm:p-6 md:p-8">
          <div className="relative w-full max-w-2xl rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 shadow-lg flex flex-col max-h-[90vh]">
            <div className="flex-none mb-4">
              <button
                className="absolute right-4 top-4 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => setIsVoiceModalOpen(false)}
              >
                <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M18 6L6 18M6 6L18 18"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <h2 className="text-lg font-semibold pr-8">Select Voice</h2>
            </div>
            <div className="flex-1 overflow-y-auto pr-1">
              <VoiceLibrary
                onClose={() => setIsVoiceModalOpen(false)}
                onSelectVoice={(voice) => {
                  setSelectedVoice(voice)
                  setIsVoiceModalOpen(false)
                }}
                initialSelectedVoice={selectedVoice}
              />
            </div>
          </div>
        </div>
      )}
    </LayoutWrapper>
  )
}
