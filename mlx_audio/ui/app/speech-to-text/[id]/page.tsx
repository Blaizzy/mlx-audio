"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { LayoutWrapper } from "@/components/layout-wrapper"
import { ArrowLeft, Download, MoreVertical, Play, Pause, SkipBack, SkipForward, ExternalLink } from "lucide-react"
import Link from "next/link"

interface Speaker {
  id: number
  name: string
  color: string
}

interface TranscriptSegment {
  speakerId: number
  startTime: number
  endTime: number
  text: string
}

export default function TranscriptViewerPage() {
  const params = useParams()
  const router = useRouter()
  const fileId = params.id as string
  const [activeTab, setActiveTab] = useState<"view" | "edit">("view")
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(8) // 8 seconds for the demo
  const audioRef = useRef<HTMLAudioElement>(null)
  const progressRef = useRef<HTMLInputElement>(null)
  const [speakers, setSpeakers] = useState<Speaker[]>([
    {
      id: 0,
      name: "Speaker 0",
      color: "#F59E0B", // Amber-500
    },
  ])

  const [transcript, setTranscript] = useState<TranscriptSegment[]>([
    {
      speakerId: 0,
      startTime: 0,
      endTime: 8,
      text: "... and the music. Shout out to Lucas Newman, Ivan Fioravanti, Andre, and Cheek Him for their amazing contributions to MLX Audio [laughs].",
    },
  ])

  const [fileName, setFileName] = useState("shoutout_000.mp3")
  const [language, setLanguage] = useState("English")
  const [date, setDate] = useState("yesterday")
  const [isExportDropdownOpen, setIsExportDropdownOpen] = useState(false)
  const [audioSource, setAudioSource] = useState<string | null>(null)

  useEffect(() => {
    // Set up audio player
    const audio = audioRef.current
    if (!audio) return

    const updateTime = () => {
      setCurrentTime(audio.currentTime)
    }

    const handleEnded = () => {
      setIsPlaying(false)
      setCurrentTime(0)
      audio.currentTime = 0
    }

    audio.addEventListener("timeupdate", updateTime)
    audio.addEventListener("ended", handleEnded)

    return () => {
      audio.removeEventListener("timeupdate", updateTime)
      audio.removeEventListener("ended", handleEnded)
    }
  }, [])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement
      if (isExportDropdownOpen && !target.closest(".export-dropdown")) {
        setIsExportDropdownOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [isExportDropdownOpen])

  useEffect(() => {
    const audio = audioRef.current
    if (audio && audioSource) {
      audio.src = audioSource
      const handleLoadedMetadata = () => {
        const metaDuration = audio.duration
        if (!Number.isNaN(metaDuration) && metaDuration > 0) {
          setDuration(metaDuration)
        }
      }
      audio.addEventListener("loadedmetadata", handleLoadedMetadata)
      return () => {
        audio.removeEventListener("loadedmetadata", handleLoadedMetadata)
      }
    } else if (audio && !audioSource) {
      audio.src = ""
    }
    return undefined
  }, [audioSource])

  const playAudio = async () => {
    const audio = audioRef.current
    if (!audio) return
    try {
      await audio.play()
      setIsPlaying(true)
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return
      }
      console.error("Error playing audio:", error)
    }
  }

  const togglePlayPause = () => {
    const audio = audioRef.current
    if (!audio || !audioSource) {
      console.warn("No audio available for playback.")
      return
    }

    if (isPlaying) {
      audio.pause()
      setIsPlaying(false)
    } else {
      void playAudio()
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = Number.parseFloat(e.target.value)
    setCurrentTime(time)
    if (audioRef.current) {
      audioRef.current.currentTime = time
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const skipBackward = () => {
    if (audioRef.current) {
      const newTime = Math.max(0, currentTime - 5)
      audioRef.current.currentTime = newTime
      setCurrentTime(newTime)
    }
  }

  const skipForward = () => {
    if (audioRef.current) {
      const newTime = Math.min(duration, currentTime + 5)
      audioRef.current.currentTime = newTime
      setCurrentTime(newTime)
    }
  }


  useEffect(() => {
    const stored = localStorage.getItem(`mlx-audio-transcription-${fileId}`)
    if (stored) {
      try {
        const data = JSON.parse(stored) as {
          fileName?: string
          language?: string
          date?: string
          duration?: number
          segments?: Array<{
            start: number
            end: number
            text: string
          }>
          text?: string
        }

        setFileName(data.fileName ?? "unknown file")
        setLanguage(data.language ?? "English")
        setDate(data.date ?? "yesterday")
        setDuration(data.duration ?? 8)
        if (data.audioDataUrl) {
          setAudioSource(data.audioDataUrl)
        } else {
          setAudioSource(null)
        }

        if (data.segments?.length) {
          const segments = data.segments.map(seg => ({
            speakerId: 0,
            startTime: seg.start,
            endTime: seg.end,
            text: seg.text
          }))
          setTranscript(segments)
        } else if (data.text) {
          setTranscript([{
            speakerId: 0,
            startTime: 0,
            endTime: data.duration ?? 8,
            text: data.text
          }])
        }
      } catch (e) {
        console.error("Error parsing stored transcription:", e)
      }
    }
  }, [fileId])


  return (
    <LayoutWrapper activeTab="audio" activePage="speech-to-text">
      <div className="flex flex-col h-full">
        {/* Top Navigation */}
        <div className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/speech-to-text">
              <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full">
                <ArrowLeft className="h-5 w-5" />
              </button>
            </Link>
            <div className="flex space-x-2">
              <button
                className={`px-4 py-1 rounded-md ${
                  activeTab === "view"
                    ? "bg-black text-white dark:bg-white dark:text-black"
                    : "bg-transparent text-gray-700 dark:text-gray-300"
                }`}
                onClick={() => setActiveTab("view")}
              >
                View
              </button>
              <button
                className={`px-4 py-1 rounded-md ${
                  activeTab === "edit"
                    ? "bg-black text-white dark:bg-white dark:text-black"
                    : "bg-transparent text-gray-700 dark:text-gray-300"
                }`}
                onClick={() => setActiveTab("edit")}
              >
                Edit
              </button>
            </div>
          </div>

          <div className="text-center font-medium">{fileName}</div>

          <div className="flex items-center space-x-2">
            {activeTab === "edit" && (
              <button
                className="flex items-center space-x-2 bg-sky-500 hover:bg-sky-600 text-white px-3 py-1.5 rounded-md mr-2"
                onClick={() => {
                  // Save changes and switch back to view mode
                  setActiveTab("view")
                  // Here you would typically save changes to a backend
                  console.log("Changes saved:", { transcript, speakers })
                }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 mr-1"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                Save
              </button>
            )}
            <div className="relative export-dropdown">
              <button
                className="flex items-center space-x-2 bg-black text-white dark:bg-white dark:text-black px-3 py-1.5 rounded-md"
                onClick={() => setIsExportDropdownOpen(!isExportDropdownOpen)}
              >
                <ExternalLink className="h-4 w-4 mr-1" />
                Export
              </button>

              {isExportDropdownOpen && (
                <div className="absolute right-0 mt-2 w-40 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-10">
                  <div className="py-1">
                    <button
                      className="w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 text-left"
                      onClick={() => {
                        console.log("Exporting as TXT")
                        setIsExportDropdownOpen(false)
                      }}
                    >
                      TXT
                    </button>
                    <button
                      className="w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 text-left"
                      onClick={() => {
                        console.log("Exporting as SRT")
                        setIsExportDropdownOpen(false)
                      }}
                    >
                      SRT
                    </button>
                    <button
                      className="w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 text-left"
                      onClick={() => {
                        console.log("Exporting as VTT")
                        setIsExportDropdownOpen(false)
                      }}
                    >
                      VTT
                    </button>
                    <button
                      className="w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 text-left"
                      onClick={() => {
                        console.log("Exporting as JSON")
                        setIsExportDropdownOpen(false)
                      }}
                    >
                      JSON
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Speakers Sidebar */}
          <div className="w-64 border-r border-gray-200 dark:border-gray-700 p-4 overflow-y-auto">
            <h2 className="text-lg font-semibold mb-4">Speakers</h2>
            {speakers.map((speaker) => (
              <div key={speaker.id} className="flex items-center space-x-2 mb-3">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs"
                  style={{ backgroundColor: speaker.color }}
                >
                  {speaker.id}
                </div>
                <span>{speaker.name}</span>
              </div>
            ))}
          </div>

          {/* Transcript Content */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-3xl mx-auto">
              <h1 className="text-2xl font-bold mb-2">{fileName}</h1>
              <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-6">
                <span>{language}</span>
                <span className="mx-2">•</span>
                <span>{date}</span>
              </div>

              <div className="mb-6 text-sm text-gray-500 dark:text-gray-400">
                {formatTime(0)} - {formatTime(duration)}
              </div>

              {transcript.map((segment, index) => {
                const speaker = speakers.find((s) => s.id === segment.speakerId)
                return (
                  <div key={index} className="mb-8">
                    <div className="flex items-start mb-2">
                      <div
                        className="w-8 h-8 rounded-full flex-shrink-0 mr-3 flex items-center justify-center text-white"
                        style={{ backgroundColor: speaker?.color || "#9CA3AF" }}
                      >
                        {speaker?.id}
                      </div>
                      <div className="flex-1">
                        {activeTab === "edit" ? (
                          <>
                            <input
                              type="text"
                              value={speaker?.name || "Unknown Speaker"}
                              onChange={(e) => {
                                const updatedSpeakers = [...speakers]
                                const speakerIndex = updatedSpeakers.findIndex((s) => s.id === segment.speakerId)
                                if (speakerIndex !== -1) {
                                  updatedSpeakers[speakerIndex] = {
                                    ...updatedSpeakers[speakerIndex],
                                    name: e.target.value,
                                  }
                                  setSpeakers(updatedSpeakers)
                                }
                              }}
                              className="font-medium mb-1 border-b border-gray-300 dark:border-gray-600 focus:border-sky-500 focus:outline-none bg-transparent"
                            />
                            <div className="relative pl-3">
                              <div
                                className="absolute left-0 top-0 bottom-0 w-1 rounded-full"
                                style={{ backgroundColor: speaker?.color || "#9CA3AF" }}
                              ></div>
                              <div
                                contentEditable
                                suppressContentEditableWarning
                                onBlur={(e) => {
                                  const updatedTranscript = [...transcript]
                                  updatedTranscript[index] = {
                                    ...updatedTranscript[index],
                                    text: e.target.innerText,
                                  }
                                  setTranscript(updatedTranscript)
                                }}
                                className="w-full text-gray-700 dark:text-gray-300 leading-relaxed focus:outline-none min-h-[3em]"
                              >
                                {segment.text}
                              </div>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="font-medium mb-1">{speaker?.name || "Unknown Speaker"}</div>
                            <div className="relative pl-3">
                              <div
                                className="absolute left-0 top-0 bottom-0 w-1 rounded-full"
                                style={{ backgroundColor: speaker?.color || "#9CA3AF" }}
                              ></div>
                              <p className="text-gray-700 dark:text-gray-300 leading-relaxed">{segment.text}</p>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Audio Player */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-2 bg-white dark:bg-gray-900">
          <div className="flex items-center justify-between">
            <div className="w-48 pl-4">{fileName}</div>

            <div className="flex items-center space-x-4">
              <button onClick={skipBackward} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full">
                <SkipBack className="h-5 w-5" />
              </button>

              <button
                onClick={togglePlayPause}
                disabled={!audioSource}
                className={`p-2 rounded-full text-white ${
                  audioSource
                    ? "bg-sky-500 hover:bg-sky-600 dark:bg-sky-500 dark:hover:bg-sky-600"
                    : "bg-gray-400 cursor-not-allowed"
                }`}
              >
                {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
              </button>

              <button onClick={skipForward} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full">
                <SkipForward className="h-5 w-5" />
              </button>
            </div>

            <div className="flex items-center w-48 justify-end space-x-2 pr-4">
              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full">
                <Download className="h-5 w-5" />
              </button>
              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full">
                <MoreVertical className="h-5 w-5" />
              </button>
            </div>
          </div>

          <div className="flex items-center px-4 mt-1">
            <span className="text-xs text-gray-500 dark:text-gray-400 w-10">{formatTime(currentTime)}</span>
            <input
              ref={progressRef}
              type="range"
              min="0"
              max={duration}
              step="0.01"
              value={currentTime}
              onChange={handleSeek}
              className="flex-1 mx-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #0ea5e9 0%, #0ea5e9 ${(currentTime / duration) * 100}%, #e5e7eb ${(currentTime / duration) * 100}%, #e5e7eb 100%)`,
              }}
            />
            <span className="text-xs text-gray-500 dark:text-gray-400 w-10 text-right">{formatTime(duration)}</span>
          </div>
        </div>

        {/* Hidden audio element */}
        <audio ref={audioRef} className="hidden">
          <source src="/placeholder-audio.mp3" type="audio/mpeg" />
          Your browser does not support the audio element.
        </audio>
      </div>
    </LayoutWrapper>
  )
}
