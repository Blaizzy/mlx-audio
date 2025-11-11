"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Ensure component is mounted to avoid hydration mismatch
  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <button className="rounded-full p-1 hover:bg-gray-100 dark:hover:bg-gray-800">
        <Sun className="h-5 w-5" />
      </button>
    )
  }

  // Force theme to be either "light" or "dark" (not "system")
  const toggleTheme = () => {
    console.log("Current theme:", resolvedTheme)
    const newTheme = resolvedTheme === "dark" ? "light" : "dark"
    console.log("Setting theme to:", newTheme)
    setTheme(newTheme)
  }

  return (
    <button
      onClick={toggleTheme}
      className="rounded-full p-1 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      aria-label="Toggle theme"
    >
      {resolvedTheme === "dark" ? <Sun className="h-5 w-5 text-gray-100" /> : <Moon className="h-5 w-5" />}
    </button>
  )
}
