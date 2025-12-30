"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { LayoutWrapper } from "@/components/layout-wrapper"
import { Upload, Play, Pause, Download, RefreshCw, ChevronDown, Volume2, VolumeX } from "lucide-react"

interface AudioTrack {
  name: string
  url: string | null
  color: string
  muted: boolean
}

export default function AudioSeparationPage() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadedAudioUrl, setUploadedAudioUrl] = useState<string | null>(null)
  const [description, setDescription] = useState("speech")
  const [method, setMethod] = useState("midpoint")
  const [steps, setSteps] = useState(16)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [tracks, setTracks] = useState<AudioTrack[]>([
    { name: "Original", url: null, color: "bg-teal-500", muted: false },
    { name: "Target (Isolated)", url: null, color: "bg-pink-500", muted: false },
    { name: "Residual (Background)", url: null, color: "bg-blue-500", muted: false },
  ])

  const audioRefs = useRef<(HTMLAudioElement | null)[]>([null, null, null])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0]
      setUploadedFile(file)
      const url = URL.createObjectURL(file)
      setUploadedAudioUrl(url)
      setTracks(prev => [
        { ...prev[0], url },
        { ...prev[1], url: null },
        { ...prev[2], url: null },
      ])
      setError(null)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      if (file.type.startsWith("audio/")) {
        setUploadedFile(file)
        const url = URL.createObjectURL(file)
        setUploadedAudioUrl(url)
        setTracks(prev => [
          { ...prev[0], url },
          { ...prev[1], url: null },
          { ...prev[2], url: null },
        ])
        setError(null)
      }
    }
  }

  const handleSeparate = async () => {
    if (!uploadedFile) return

    setIsProcessing(true)
    setError(null)

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost'
    const API_PORT = process.env.NEXT_PUBLIC_API_PORT || '8000'

    const formData = new FormData()
    formData.append("file", uploadedFile)
    formData.append("model", "facebook/sam-audio-large")
    formData.append("description", description)
    formData.append("method", method)
    formData.append("steps", steps.toString())

    try {
      const response = await fetch(`${API_BASE_URL}:${API_PORT}/v1/audio/separations`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Convert base64 to blob URLs
      const targetBlob = base64ToBlob(data.target, "audio/wav")
      const residualBlob = base64ToBlob(data.residual, "audio/wav")

      const targetUrl = URL.createObjectURL(targetBlob)
      const residualUrl = URL.createObjectURL(residualBlob)

      setTracks(prev => [
        prev[0],
        { ...prev[1], url: targetUrl },
        { ...prev[2], url: residualUrl },
      ])
    } catch (err) {
      console.error("Error separating audio:", err)
      setError(err instanceof Error ? err.message : "Failed to separate audio")
    } finally {
      setIsProcessing(false)
    }
  }

  const base64ToBlob = (base64: string, mimeType: string): Blob => {
    const byteCharacters = atob(base64)
    const byteNumbers = new Array(byteCharacters.length)
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }
    const byteArray = new Uint8Array(byteNumbers)
    return new Blob([byteArray], { type: mimeType })
  }

  const handlePlayPause = () => {
    if (isPlaying) {
      audioRefs.current.forEach(audio => audio?.pause())
      setIsPlaying(false)
    } else {
      // Only play non-muted tracks that have audio
      audioRefs.current.forEach((audio, idx) => {
        if (audio && tracks[idx].url && !tracks[idx].muted) {
          audio.currentTime = 0
          audio.play()
        }
      })
      setIsPlaying(true)
    }
  }

  const toggleMute = (index: number) => {
    setTracks(prev => prev.map((track, i) =>
      i === index ? { ...track, muted: !track.muted } : track
    ))

    const audio = audioRefs.current[index]
    if (audio) {
      audio.muted = !tracks[index].muted
    }
  }

  const handleDownload = (index: number) => {
    const track = tracks[index]
    if (!track.url) return

    const a = document.createElement("a")
    a.href = track.url
    a.download = `${track.name.toLowerCase().replace(/[^a-z0-9]/g, "_")}.wav`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleStartOver = () => {
    // Stop all audio
    audioRefs.current.forEach(audio => {
      if (audio) {
        audio.pause()
        audio.currentTime = 0
      }
    })
    setIsPlaying(false)

    // Revoke URLs
    tracks.forEach(track => {
      if (track.url) URL.revokeObjectURL(track.url)
    })

    // Reset state
    setUploadedFile(null)
    setUploadedAudioUrl(null)
    setTracks([
      { name: "Original", url: null, color: "bg-teal-500", muted: false },
      { name: "Target (Isolated)", url: null, color: "bg-pink-500", muted: false },
      { name: "Residual (Background)", url: null, color: "bg-blue-500", muted: false },
    ])
    setError(null)
  }

  // Sync audio playback
  useEffect(() => {
    const handleEnded = () => setIsPlaying(false)

    audioRefs.current.forEach(audio => {
      if (audio) {
        audio.addEventListener("ended", handleEnded)
      }
    })

    return () => {
      audioRefs.current.forEach(audio => {
        if (audio) {
          audio.removeEventListener("ended", handleEnded)
        }
      })
    }
  }, [tracks])

  const hasResults = tracks[1].url !== null && tracks[2].url !== null

  return (
    <LayoutWrapper activeTab="audio" activePage="audio-separation">
      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel - Settings */}
        <div className="w-80 border-r border-gray-200 dark:border-gray-700 p-6 overflow-auto">
          <h1 className="text-xl font-bold mb-2">Audio Separation</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
            Extract sounds from audio using SAM Audio
          </p>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mb-6">
            <h2 className="text-sm font-medium mb-3">How it works</h2>
            <ol className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li className="flex items-start">
                <span className="font-medium mr-2">1.</span>
                <span>Upload audio file</span>
              </li>
              <li className="flex items-start">
                <span className="font-medium mr-2">2.</span>
                <span>Describe what to isolate</span>
              </li>
              <li className="flex items-start">
                <span className="font-medium mr-2">3.</span>
                <span>Get target and residual</span>
              </li>
            </ol>
          </div>

          {/* Description Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">What to isolate</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g., speech, guitar, drums"
              className="w-full rounded-lg border border-gray-200 dark:border-gray-700 px-3 py-2 text-sm bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Describe the sound you want to separate
            </p>
          </div>

          {/* Method Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">Method</label>
            <div className="relative">
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="w-full appearance-none rounded-lg border border-gray-200 dark:border-gray-700 px-3 py-2 pr-8 text-sm bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500"
              >
                <option value="midpoint">Midpoint (Higher Quality)</option>
                <option value="euler">Euler (Faster)</option>
              </select>
              <ChevronDown className="absolute right-2 top-2.5 h-4 w-4 text-gray-400 pointer-events-none" />
            </div>
          </div>

          {/* Steps Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">Steps: {steps}</label>
            <div className="flex space-x-2">
              {[2, 4, 8, 16, 32].map((s) => (
                <button
                  key={s}
                  onClick={() => setSteps(s)}
                  className={`px-3 py-1 text-xs rounded-md ${
                    steps === s
                      ? "bg-sky-500 text-white"
                      : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              More steps = higher quality, slower
            </p>
          </div>

          {/* Separate Button */}
          {uploadedFile && (
            <button
              onClick={handleSeparate}
              disabled={isProcessing || !uploadedFile}
              className={`w-full rounded-lg py-3 text-sm font-medium text-white ${
                isProcessing
                  ? "bg-sky-400 cursor-not-allowed"
                  : "bg-sky-500 hover:bg-sky-600"
              } flex items-center justify-center`}
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Separating...
                </>
              ) : (
                "Separate Audio"
              )}
            </button>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Right Panel - Upload & Results */}
        <div className="flex-1 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-950">
          {!uploadedFile ? (
            /* Upload Area */
            <div className="flex-1 flex items-center justify-center p-8">
              <div
                className={`w-full max-w-xl border-2 border-dashed rounded-xl p-12 text-center ${
                  isDragging
                    ? "border-sky-500 bg-sky-50 dark:bg-sky-900/20"
                    : "border-gray-300 dark:border-gray-700"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="mx-auto w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4">
                  <Upload className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium mb-2">Upload audio file</h3>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  Click to upload or drag and drop
                </p>
                <p className="text-xs text-gray-400 dark:text-gray-500">
                  Supports MP3, WAV, FLAC, and other audio formats
                </p>
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="audio/*"
                  onChange={handleFileUpload}
                />
              </div>
            </div>
          ) : (
            /* Results View */
            <div className="flex-1 flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-3">
                  <span className="text-sm font-medium">{uploadedFile.name}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleStartOver}
                    className="px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800"
                  >
                    Start Over
                  </button>
                </div>
              </div>

              {/* Audio Visualization Area */}
              <div className="flex-1 flex items-center justify-center bg-gradient-to-b from-gray-900 to-black p-8">
                <div className="w-full h-24 bg-gradient-to-r from-pink-500/20 via-purple-500/30 to-pink-500/20 rounded-lg" />
              </div>

              {/* Playback Controls */}
              <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
                <div className="flex items-center justify-center mb-4">
                  <button
                    onClick={handlePlayPause}
                    disabled={!hasResults && !uploadedAudioUrl}
                    className="w-12 h-12 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isPlaying ? (
                      <Pause className="h-6 w-6" />
                    ) : (
                      <Play className="h-6 w-6 ml-0.5" />
                    )}
                  </button>
                </div>

                {/* Track List */}
                <div className="space-y-3">
                  {tracks.map((track, index) => (
                    <div
                      key={track.name}
                      className={`flex items-center space-x-3 p-3 rounded-lg ${
                        track.url
                          ? "bg-gray-50 dark:bg-gray-800"
                          : "bg-gray-50/50 dark:bg-gray-800/50 opacity-50"
                      }`}
                    >
                      <button
                        onClick={() => toggleMute(index)}
                        disabled={!track.url}
                        className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 disabled:cursor-not-allowed"
                      >
                        {track.muted ? (
                          <VolumeX className="h-4 w-4 text-gray-400" />
                        ) : (
                          <Volume2 className="h-4 w-4" />
                        )}
                      </button>

                      <div className={`w-1 h-8 rounded-full ${track.color}`} />

                      <div className="flex-1">
                        <span className="text-sm font-medium">{track.name}</span>
                        {track.url && (
                          <div className="h-6 mt-1 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                            <div className={`h-full ${track.color} opacity-60`} style={{ width: "100%" }} />
                          </div>
                        )}
                      </div>

                      <button
                        onClick={() => handleDownload(index)}
                        disabled={!track.url}
                        className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <Download className="h-4 w-4" />
                      </button>

                      {/* Hidden audio element */}
                      <audio
                        ref={(el) => { audioRefs.current[index] = el }}
                        src={track.url || undefined}
                        muted={track.muted}
                      />
                    </div>
                  ))}
                </div>

                {!hasResults && uploadedFile && (
                  <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-4">
                    Click &quot;Separate Audio&quot; to generate target and residual tracks
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </LayoutWrapper>
  )
}
