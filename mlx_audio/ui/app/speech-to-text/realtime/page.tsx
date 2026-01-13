"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { LayoutWrapper } from "@/components/layout-wrapper"
import { Mic, MicOff, ChevronDown } from "lucide-react"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export default function RealtimeTranscriptionPage() {
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState<string>("")
  const [language, setLanguage] = useState("Detect")
  const [selectedModel, setSelectedModel] = useState("mlx-community/whisper-large-v3-turbo-asr-fp16")
  const [status, setStatus] = useState<"idle" | "connecting" | "ready" | "recording" | "error">("idle")
  const [error, setError] = useState<string | null>(null)
  const [isSpeechDetected, setIsSpeechDetected] = useState(false)

  const websocketRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)

  const startAudioProcessing = (
    stream: MediaStream,
    audioContext: AudioContext,
    ws: WebSocket
  ) => {
    console.log("Starting audio processing, sample rate:", audioContext.sampleRate)
    const source = audioContext.createMediaStreamSource(stream)

    // Create a ScriptProcessorNode for processing audio
    // Use buffer size that works well with 16kHz target
    const bufferSize = 4096
    const processor = audioContext.createScriptProcessor(bufferSize, 1, 1)
    processorRef.current = processor

    let sampleCount = 0

    processor.onaudioprocess = (event) => {
      if (ws.readyState === WebSocket.OPEN) {
        const inputData = event.inputBuffer.getChannelData(0)
        sampleCount += inputData.length

        // Resample if needed (browser usually uses 48kHz, we need 16kHz)
        const targetSampleRate = 16000
        const sourceSampleRate = audioContext.sampleRate
        let processedData = inputData

        if (sourceSampleRate !== targetSampleRate) {
          // Simple downsampling: take every Nth sample
          const ratio = sourceSampleRate / targetSampleRate
          const newLength = Math.floor(inputData.length / ratio)
          processedData = new Float32Array(newLength)
          for (let i = 0; i < newLength; i++) {
            processedData[i] = inputData[Math.floor(i * ratio)]
          }
        }

        // Convert float32 to int16
        const int16Array = new Int16Array(processedData.length)
        for (let i = 0; i < processedData.length; i++) {
          const s = Math.max(-1, Math.min(1, processedData[i]))
          int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff
        }

        // Send audio data
        try {
          ws.send(int16Array.buffer)
          if (sampleCount % (sourceSampleRate * 2) < bufferSize) {
            // Log every ~2 seconds
            console.log(`Sent ${sampleCount} samples (${(sampleCount / sourceSampleRate).toFixed(2)}s)`)
          }
        } catch (error) {
          console.error("Error sending audio data:", error)
        }
      }
    }

    source.connect(processor)
    // Connect processor to a dummy destination to ensure it processes
    // We'll use a GainNode with gain 0 to avoid audio output
    const gainNode = audioContext.createGain()
    gainNode.gain.value = 0
    processor.connect(gainNode)
    gainNode.connect(audioContext.destination)

    console.log("Audio processing started")
  }

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      stopRecording()
    }
  }, [])

  const startRecording = async () => {
    try {
      setError(null)
      setStatus("connecting")

      // Get microphone access with noise suppression and echo cancellation
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000, // Request 16kHz if supported
        },
      })
      streamRef.current = stream

      // Apply additional constraints for better noise suppression
      const audioTrack = stream.getAudioTracks()[0]
      if (audioTrack && typeof audioTrack.getCapabilities === 'function') {
        try {
          await audioTrack.applyConstraints({
            advanced: [
              { noiseSuppression: true },
              { echoCancellation: true },
              { autoGainControl: true },
            ],
          })
          console.log("Noise suppression and echo cancellation enabled")
        } catch (err) {
          console.warn("Could not apply audio constraints:", err)
        }
      }

      // Create audio context (browser may not support custom sample rates)
      // We'll handle resampling in the processor
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      audioContextRef.current = audioContext
      console.log("Audio context created with sample rate:", audioContext.sampleRate)

      // Connect to WebSocket first
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost"
      const API_PORT = process.env.NEXT_PUBLIC_API_PORT || "8000"
      const wsProtocol = API_BASE_URL.startsWith("https") ? "wss" : "ws"
      const host = API_BASE_URL.replace(/^https?:\/\//, "").replace(/^http:\/\//, "").split(":")[0]
      const wsUrl = `${wsProtocol}://${host}:${API_PORT}/v1/audio/transcriptions/realtime`

      const ws = new WebSocket(wsUrl)
      websocketRef.current = ws

      ws.onopen = () => {
        // Send configuration
        ws.send(
          JSON.stringify({
            model: selectedModel,
            language: language === "Detect" ? null : language.toLowerCase(),
            sample_rate: 16000,
          })
        )
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.status === "ready") {
            setStatus("ready")
            setIsRecording(true)
            setStatus("recording")

            // Start processing audio stream
            startAudioProcessing(stream, audioContext, ws)
          } else if (data.text) {
            // Handle transcription (partial or final)
            const isPartial = data.is_partial || false

            if (isPartial) {
              // Partial transcription - append with a marker that it might be updated
              setTranscript((prev) => {
                const partialText = data.text.trim()
                if (!prev || prev.trim().length === 0) {
                  return partialText
                }
                // Append partial text (it will be replaced by final transcription)
                if (prev && !prev.endsWith(" ") && partialText) {
                  return prev + " " + partialText
                }
                return prev + (partialText || "")
              })
            } else {
              // Final transcription - replace the last partial if exists, or append
              setTranscript((prev) => {
                const finalText = data.text.trim()
                if (!prev || prev.trim().length === 0) {
                  return finalText
                }

                // If we have a previous partial transcription, try to replace it
                // Estimate: partial is usually ~1.5 seconds, roughly 3-4 words
                const words = prev.trim().split(/\s+/).filter(w => w.length > 0)
                const estimatedPartialWords = 4

                if (words.length >= estimatedPartialWords) {
                  // Replace last few words (likely the partial) with final transcription
                  const wordsToKeep = words.slice(0, Math.max(0, words.length - estimatedPartialWords))
                  const newText = wordsToKeep.length > 0
                    ? wordsToKeep.join(" ") + " " + finalText
                    : finalText
                  return newText
                } else {
                  // Just append if transcript is short
                  if (prev && !prev.endsWith(" ") && finalText) {
                    return prev + " " + finalText
                  }
                  return prev + (finalText || "")
                }
              })
            }

            setIsSpeechDetected(true)
            // Reset speech indicator after a delay
            setTimeout(() => setIsSpeechDetected(false), 100)
          } else if (data.error) {
            setError(data.error)
            setStatus("error")
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e)
        }
      }

      ws.onerror = (error) => {
        console.error("WebSocket error:", error)
        setError("Connection error. Please check if the server is running.")
        setStatus("error")
      }

      ws.onclose = () => {
        setStatus("idle")
        setIsRecording(false)
      }
    } catch (err: any) {
      console.error("Error starting recording:", err)
      setError(err.message || "Failed to start recording. Please check microphone permissions.")
      setStatus("error")
    }
  }

  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    if (websocketRef.current) {
      if (websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ action: "stop" }))
      }
      websocketRef.current.close()
      websocketRef.current = null
    }

    setIsRecording(false)
    setStatus("idle")
  }

  const clearTranscript = () => {
    setTranscript("")
  }

  return (
    <LayoutWrapper activeTab="audio" activePage="speech-to-text">
      <div className="flex-1 overflow-auto p-6">
        <div className="flex items-center mb-6">
          <Link href="/speech-to-text">
            <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full mr-4">
              <ArrowLeft className="h-5 w-5" />
            </button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">Realtime Transcription</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Experience realtime speech transcription with our latest model.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Control Panel */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-semibold mb-2">Control Panel</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
              Manage your transcription session.
            </p>

            <div className="space-y-6">
              {/* Language Selection */}
              <div>
                <label className="block text-sm font-medium mb-2">Language</label>
                <div className="relative">
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    disabled={isRecording}
                    className="w-full appearance-none rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2.5 pr-10 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
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

              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium mb-2">Model</label>
                <input
                  type="text"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={isRecording}
                  placeholder="Enter model name"
                  className="w-full rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2.5 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>

              {/* Error Display */}
              {error && (
                <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
              )}

              {/* Control Button */}
              <div className="pt-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={status === "connecting"}
                    className="w-full flex items-center justify-center space-x-2 bg-sky-500 hover:bg-sky-600 text-white px-6 py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Mic className="h-5 w-5" />
                    <span>{status === "connecting" ? "Connecting..." : "Start Transcribing"}</span>
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="w-full flex items-center justify-center space-x-2 bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg transition-colors"
                  >
                    <MicOff className="h-5 w-5" />
                    <span>Stop</span>
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Live Transcript */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Live Transcript</h2>
              {isRecording && (
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${
                    isSpeechDetected ? "bg-green-500" : "bg-red-500"
                  }`}></div>
                  <span className={`text-sm ${
                    isSpeechDetected ? "text-green-500" : "text-red-500"
                  }`}>
                    {isSpeechDetected ? "Speech Detected" : "Listening..."}
                  </span>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto min-h-[400px] max-h-[600px] p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
              {transcript ? (
                <p className="text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">
                  {transcript}
                </p>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <div className="mb-4 p-4 bg-gray-200 dark:bg-gray-700 rounded-full">
                    <Mic className="h-8 w-8 text-gray-400" />
                  </div>
                  <p className="text-gray-500 dark:text-gray-400 mb-2">
                    Ready to transcribe your speech...
                  </p>
                  <p className="text-sm text-gray-400 dark:text-gray-500">
                    Click &apos;Start Transcribing&apos; to begin.
                  </p>
                </div>
              )}
            </div>

            {transcript && (
              <div className="mt-4 flex justify-end">
                <button
                  onClick={clearTranscript}
                  className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                >
                  Clear
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </LayoutWrapper>
  )
}

