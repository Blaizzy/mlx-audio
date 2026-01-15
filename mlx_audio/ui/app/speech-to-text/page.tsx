"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { LayoutWrapper } from "@/components/layout-wrapper"
import { FileText, Upload, MoreVertical, X, ChevronDown, Mic } from "lucide-react"
import Link from "next/link"

interface TranscriptionFile {
  id: string
  name: string
  status: "uploading" | "processing" | "completed" | "failed"
  result?: string
}
export default function SpeechToTextPage() {
  const [files, setFiles] = useState<TranscriptionFile[]>([
    {
      id: "1",
      name: "shoutout_000.mp3",
      status: "completed",
    },
  ])
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [primaryLanguage, setPrimaryLanguage] = useState("Detect")
  const [tagAudioEvents, setTagAudioEvents] = useState(false)
  const [selectedModel, setSelectedModel] = useState("mlx-community/whisper-large-v3-turbo-asr-fp16")
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  // New state for stored transcriptions
  const [storedTranscriptions, setStoredTranscriptions] = useState<{ id: string; data: any }[]>([])

  // Function to load transcriptions from localStorage
  const loadStoredTranscriptions = () => {
    const keys = Object.keys(localStorage).filter((key) => key.startsWith("mlx-audio-transcription-"))
    const transcriptions = keys.map((key) => {
      const dataStr = localStorage.getItem(key)
      try {
        const data = dataStr ? JSON.parse(dataStr) : {}
        return {
          id: key.replace("mlx-audio-transcription-", ""),
          data,
          name: data.fileName || "Unknown file",
          status: "completed" as const
        }
      } catch (error) {
        return {
          id: key.replace("mlx-audio-transcription-", ""),
          data: {},
          name: "Unknown file",
          status: "completed" as const
        }
      }
    })

    // Keep the default file and add stored transcriptions
    setFiles(prev => [
      prev[0],
      ...transcriptions.map(t => ({
        id: t.id,
        name: t.name,
        status: t.status
      }))
    ])
    setStoredTranscriptions(transcriptions)
  }

  useEffect(() => {
    loadStoredTranscriptions()
  }, [])

  const uploadAndTranscribe = async (file: File, id: string) => {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("model", selectedModel)
    formData.append("language", primaryLanguage === "Detect" ? "en" : primaryLanguage.toLowerCase())
    formData.append("response_format", "verbose_json")
    formData.append("temperature", "0")

    let fileName = file.name

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost';
    const API_PORT = process.env.NEXT_PUBLIC_API_PORT || '8000';

    try {
      const res = await fetch(`${API_BASE_URL}:${API_PORT}/v1/audio/transcriptions`, {
        method: "POST",
        body: formData,
      })
      const data = await res.json()

      // Save the transcription to a JSON file via API
      data.fileName = fileName
      localStorage.setItem(`mlx-audio-transcription-${id}`, JSON.stringify(data))
      setFiles((prev) =>
        prev.map((f) =>
          f.id === id ? { ...f, status: "completed", result: data.text } : f
        )
      )
    } catch (err) {
      setFiles((prev) =>
        prev.map((f) => (f.id === id ? { ...f, status: "failed" } : f))
      )
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newEntries: TranscriptionFile[] = Array.from(e.target.files).map((file) => ({
        id: Math.random().toString(36).substring(2, 9),
        name: file.name,
        status: "uploading",
      }))

      setFiles([...files, ...newEntries])
      setIsModalOpen(false)

      newEntries.forEach((entry, idx) => {
        const file = e.target.files![idx]
        uploadAndTranscribe(file, entry.id)
      })
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
      const newEntries: TranscriptionFile[] = Array.from(e.dataTransfer.files).map((file) => ({
        id: Math.random().toString(36).substring(2, 9),
        name: file.name,
        status: "uploading",
      }))

      setFiles([...files, ...newEntries])
      setIsModalOpen(false)

      newEntries.forEach((entry, idx) => {
        const file = e.dataTransfer.files[idx]
        uploadAndTranscribe(file, entry.id)
      })
    }
  }

  const deleteFile = (id: string) => {
    setFiles(files.filter((file) => file.id !== id))
    localStorage.removeItem(`mlx-audio-transcription-${id}`)
  }

  return (
    <LayoutWrapper activeTab="audio" activePage="speech-to-text">
      <div className="flex-1 overflow-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold">Speech to text</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Transcribe audio and video files with our{" "}
              <span className="text-gray-700 dark:text-gray-300 hover:underline cursor-pointer">
                industry-leading ASR model
              </span>
              .
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <Link href="/speech-to-text/realtime">
              <button className="flex items-center space-x-2 bg-sky-500 hover:bg-sky-600 text-white px-4 py-2 rounded-lg transition-colors">
                <Mic className="h-5 w-5" />
                <span>Realtime Transcription</span>
              </button>
            </Link>
            <button
              onClick={() => setIsModalOpen(true)}
              className="flex items-center space-x-2 bg-black dark:bg-white text-white dark:text-black px-4 py-2 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors"
            >
              <FileText className="h-5 w-5" />
              <span>Transcribe files</span>
            </button>
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
          {files.length > 0 ? (
            <div className="space-y-4">
              {files.map((file) => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center">
                    <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mr-4">
                      <FileText className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                    </div>
                    <div>
                      <h3 className="font-medium">{file.name}</h3>
                      {file.status === "uploading" && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Uploading...</p>
                      )}
                      {file.status === "processing" && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Processing...</p>
                      )}
                      {file.status === "completed" && (
                        <p className="text-xs text-green-500 dark:text-green-400 mt-1">Completed</p>
                      )}
                      {file.status === "failed" && (
                        <p className="text-xs text-red-500 dark:text-red-400 mt-1">Failed</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center">
                    {file.status === "completed" && (
                      <Link href={`/speech-to-text/${file.id}`}>
                        <button className="text-sky-500 hover:text-sky-600 text-sm mr-4">View transcript</button>
                      </Link>
                    )}
                    <div className="relative group">
                      <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700">
                        <MoreVertical className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                      </button>
                      <div className="absolute right-0 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 hidden group-hover:block z-10">
                        <div className="py-1">
                          {file.status === "completed" && (
                            <>
                              <button className="flex items-center w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                                Download transcript
                              </button>
                              <button className="flex items-center w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                                Copy to clipboard
                              </button>
                            </>
                          )}
                          <button
                            className="flex items-center w-full px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                            onClick={() => deleteFile(file.id)}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="mx-auto w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4">
                <FileText className="h-8 w-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium mb-2">No transcriptions yet</h3>
              <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md mx-auto">
                Upload audio or video files to transcribe them into text using our advanced speech recognition
                technology.
              </p>
              <button
                onClick={() => setIsModalOpen(true)}
                className="bg-sky-500 hover:bg-sky-600 text-white px-4 py-2 rounded-md transition-colors"
              >
                Transcribe your first file
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Transcribe Files Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="relative w-full max-w-2xl bg-white dark:bg-gray-900 rounded-xl p-6">
            <button
              onClick={() => setIsModalOpen(false)}
              className="absolute right-6 top-6 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
            >
              <X className="h-6 w-6" />
            </button>

            <h2 className="text-2xl font-bold mb-6">Transcribe files</h2>

            <div
              className={`border ${
                isDragging ? "border-sky-500 dark:border-sky-400" : "border-dashed border-gray-300 dark:border-gray-600"
              } rounded-lg p-8 mb-6`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="flex flex-col items-center justify-center text-center">
                <div className="mb-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-full">
                  <Upload className="h-6 w-6 text-gray-500 dark:text-gray-400" />
                </div>
                <h3 className="text-lg font-medium mb-2">Click to upload, or drag and drop</h3>
                <p className="text-gray-500 dark:text-gray-400 mb-2">Audio or video files, up to 1000MB each</p>
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="audio/*,video/*"
                  onChange={handleFileUpload}
                  multiple
                />
              </div>
            </div>

            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <label className="block text-sm font-medium">Primary language</label>
              </div>
              <div className="relative">
                <select
                  value={primaryLanguage}
                  onChange={(e) => setPrimaryLanguage(e.target.value)}
                  className="w-full appearance-none rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2.5 pr-10 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500"
                >
                  <option value="Detect">Detect</option>
                  <option value="English">English</option>
                  <option value="Spanish">Spanish</option>
                  <option value="French">French</option>
                  <option value="German">German</option>
                  <option value="Italian">Italian</option>
                  <option value="Portuguese">Portuguese</option>
                  <option value="Chinese">Chinese</option>
                  <option value="Japanese">Japanese</option>
                  <option value="Korean">Korean</option>
                </select>
                <ChevronDown className="absolute right-3 top-3 h-5 w-5 text-gray-400 pointer-events-none" />
              </div>
            </div>

            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <label className="block text-sm font-medium">Model</label>
              </div>
              <input
                type="text"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                placeholder="Enter model name"
                className="w-full rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2.5 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500"
              />
            </div>

            <div className="mb-6">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium">Tag audio events</label>
                <div className="relative inline-block w-12 h-6 transition duration-200 ease-in-out">
                  <input
                    type="checkbox"
                    id="toggle"
                    className="opacity-0 w-0 h-0"
                    checked={tagAudioEvents}
                    onChange={() => setTagAudioEvents(!tagAudioEvents)}
                  />
                  <label
                    htmlFor="toggle"
                    className={`absolute cursor-pointer top-0 left-0 right-0 bottom-0 rounded-full ${
                      tagAudioEvents ? "bg-black dark:bg-white" : "bg-gray-300 dark:bg-gray-600"
                    }`}
                  >
                    <span
                      className={`absolute left-1 bottom-1 bg-white dark:bg-gray-800 w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${
                        tagAudioEvents ? "transform translate-x-6" : ""
                      }`}
                    ></span>
                  </label>
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center space-x-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 px-6 py-3 rounded-lg transition-colors"
              >
                <Upload className="h-5 w-5 mr-2" />
                Upload files
              </button>
            </div>
          </div>
        </div>
      )}
    </LayoutWrapper>
  )
}
