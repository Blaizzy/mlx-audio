"use client"

import { useState, useEffect } from "react"
import { VoiceLibrary } from "@/components/voice-library"
import { Settings } from "lucide-react"

interface VoiceSelectionProps {
  onVoiceChange?: (voice: string) => void
  initialVoice?: string
  className?: string
}

export function VoiceSelection({
  onVoiceChange,
  initialVoice = "Trustworthy Man",
  className = "mb-6",
}: VoiceSelectionProps) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedVoice, setSelectedVoice] = useState(initialVoice)

  useEffect(() => {
    // This ensures the UI updates when the selected voice changes
    console.log("Selected voice updated:", selectedVoice)
  }, [selectedVoice])

  useEffect(() => {
    // Update selected voice if initialVoice prop changes
    if (initialVoice) {
      setSelectedVoice(initialVoice)
    }
  }, [initialVoice])

  // Helper function to determine gradient based on voice name
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

  const handleVoiceChange = (voice: string) => {
    setSelectedVoice(voice)
    if (onVoiceChange) {
      onVoiceChange(voice)
    }
    setIsModalOpen(false)
  }

  const handleResetVoice = () => {
    const defaultVoice = "Trustworthy Man"
    setSelectedVoice(defaultVoice)
    if (onVoiceChange) {
      onVoiceChange(defaultVoice)
    }
  }

  return (
    <div className={className}>
      <h3 className="mb-4 text-sm font-medium">Voice</h3>

      <div className="flex items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 p-2 hover:bg-gray-50 dark:hover:bg-gray-800">
        <div className="flex items-center space-x-3">
          <div
            className={`w-10 h-10 rounded-md flex-shrink-0 bg-gradient-to-br ${getGradientForVoice(selectedVoice)}`}
            aria-label={`${selectedVoice} avatar`}
          ></div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">{selectedVoice}</span>
              <div className="h-2 w-2 rounded-full bg-gray-200 dark:bg-gray-600"></div>
            </div>
            <span className="text-xs text-gray-500 dark:text-gray-400">English</span>
          </div>
        </div>
        <button
          className="rounded-md border border-gray-200 dark:border-gray-700 p-1 hover:bg-gray-100 dark:hover:bg-gray-700"
          onClick={() => setIsModalOpen(true)}
        >
          <Settings className="h-4 w-4" />
        </button>
      </div>
      <div className="mt-2 text-right">
        <button
          className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          onClick={handleResetVoice}
        >
          <span className="mr-1">Reset Voice</span>
          <svg className="inline h-3 w-3" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M9 12L11 14L15 10"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>

      {/* Voice Selection Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 sm:p-6 md:p-8">
          <div className="relative w-full max-w-2xl rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 shadow-lg flex flex-col max-h-[90vh]">
            <div className="flex-none mb-4">
              <button
                className="absolute right-4 top-4 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => setIsModalOpen(false)}
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
                onClose={() => setIsModalOpen(false)}
                onSelectVoice={(voice) => {
                  handleVoiceChange(voice)
                  setIsModalOpen(false)
                }}
                initialSelectedVoice={selectedVoice}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default VoiceSelection
