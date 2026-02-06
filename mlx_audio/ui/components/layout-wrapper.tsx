"use client"

import type { ReactNode } from "react"
import { Navbar } from "@/components/navbar"
import { Sidebar } from "@/components/sidebar"
import { Footer } from "@/components/footer"

interface LayoutWrapperProps {
  children: ReactNode
  activeTab?: "audio" | "chat" | "video"
  activePage?: "home" | "text-to-speech" | "speech-to-speech" | "voices" | "speech-to-text" | "audio-separation"
}

export function LayoutWrapper({ children, activeTab = "audio", activePage = "home" }: LayoutWrapperProps) {
  return (
    <div className="flex h-screen flex-col bg-white dark:bg-gray-900 dark:text-white transition-colors">
      <Navbar activeTab={activeTab} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar activePage={activePage} />

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">{children}</div>
      </div>

      <Footer />
    </div>
  )
}
