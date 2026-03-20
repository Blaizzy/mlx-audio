"use client"

import Link from "next/link"
import { FileAudio, FileText, AudioWaveform } from "lucide-react"

interface SidebarProps {
  activePage?: "home" | "text-to-speech" | "speech-to-speech" | "voices" | "studio" | "speech-to-text" | "audio-separation"
}

export function Sidebar({ activePage = "home" }: SidebarProps) {
  return (
    <aside className="w-48 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
      <nav className="flex flex-col space-y-1 p-3">

        <Link
          href="/text-to-speech"
          className={`flex items-center space-x-3 rounded-md px-3 py-2 text-sm ${
            activePage === "text-to-speech"
              ? "bg-gray-100 dark:bg-gray-800 font-medium text-black dark:text-white"
              : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
          }`}
        >
          <FileAudio className="h-5 w-5" />
          <span>Text to Speech</span>
        </Link>

        <Link
          href="/speech-to-text"
          className={`flex items-center space-x-3 rounded-md px-3 py-2 text-sm ${
            activePage === "speech-to-text"
              ? "bg-gray-100 dark:bg-gray-800 font-medium text-black dark:text-white"
              : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
          }`}
        >
          <FileText className="h-5 w-5" />
          <span>Speech to Text</span>
        </Link>

        <Link
          href="/audio-separation"
          className={`flex items-center space-x-3 rounded-md px-3 py-2 text-sm ${
            activePage === "audio-separation"
              ? "bg-gray-100 dark:bg-gray-800 font-medium text-black dark:text-white"
              : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
          }`}
        >
          <AudioWaveform className="h-5 w-5" />
          <span>Audio Separation</span>
        </Link>

      </nav>
    </aside>
  )
}
