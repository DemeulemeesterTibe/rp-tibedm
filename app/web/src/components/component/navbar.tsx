import Link from "next/link"


export function Navbar() {
    return (
        <nav className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 p-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold">Research Project</h1>
          <div className="flex gap-4">
            <Link className="text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400" href="/demo">
              Demo
            </Link>
            <Link className="text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400" href="/speech">
              Speech Synthesis
            </Link>
            <Link className="text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400" href="/songs">
              Songs
            </Link>
          </div>
        </div>
      </nav>
    )
}