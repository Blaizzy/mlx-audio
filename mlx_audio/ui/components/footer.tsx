import Link from "next/link"

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-700 px-4 py-2 text-xs text-gray-500 dark:text-gray-400">
      <div className="flex items-center justify-between">
        <div className="flex space-x-4">
          <Link href="#" className="hover:underline">
            About MLX-Audio
          </Link>
          <Link href="#" className="hover:underline">
            Terms of Service
          </Link>
          <Link href="#" className="hover:underline">
            Privacy Policy
          </Link>
          <Link href="#" className="hover:underline">
            @MLX-Audio 2025
          </Link>
        </div>
        <button className="flex items-center space-x-1 rounded-md border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs hover:bg-gray-50 dark:hover:bg-gray-800">
          <span>Access API</span>
        </button>
      </div>
    </footer>
  )
}
