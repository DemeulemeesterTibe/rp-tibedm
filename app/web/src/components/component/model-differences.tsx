/**
 * This code was generated by v0 by Vercel.
 * @see https://v0.dev/t/YC5pr0u6MBG
 */
import { Navbar } from "./navbar"

export function ModelDifferences() {
  return (
    <div key="1" className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <Navbar />
      <div className="flex h-full">
        <main className="flex-1 p-6 overflow-y-auto flex flex-col items-center">
          <h1 className="text-3xl font-bold mb-6 text-center">Model Differences</h1>
          <h2 className="text-2xl font-semibold mb-4 text-center">Speech Synthesis</h2>
          <div className="grid grid-cols-4 gap-8">
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Reference Audio</h3>
              <audio controls>
                <source src="/audio/modelDiff/obamaRef.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Bark</h3>
              <audio controls>
                <source src="/audio/modelDiff/obamaBark.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Tortoise (Standard)</h3>
              <audio controls>
                <source src="/audio/modelDiff/obamaTortoise.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">XTSS2</h3>
              <audio controls>
                <source src="/audio/modelDiff/obamaXtss.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/alanWakeRef.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/alanWakeBark.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/alanWakeTortoise.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/alanWakeXtss.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
          </div>
          <h2 className="text-2xl font-semibold mb-4 mt-4 text-center">Fine-Tuning a Model </h2>
          <div className="grid grid-cols-3 gap-8">
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Reference Audio</h3>
              <audio controls>
                <source src="/audio/modelDiff/obamaRef.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">XTSS2</h3>
              <audio controls>
                <source src="/audio/modelDiff/obamaXtssFineTune.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Tortoise</h3>
              <audio controls>
                <source src="/audiofile3.mp3" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/nathanRef.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/nathanXtssFineTune.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <audio controls>
                <source src="/audio/modelDiff/nathanTortoiseFineTune.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
          </div>
          <h2 className="text-2xl font-semibold mb-4 mt-4 text-center">Online models</h2>
          <div className="grid grid-cols-4 gap-8">
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Reference Audio</h3>
              <audio controls>
                <source src="/audio/modelDiff/tibeRef.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Play.ht</h3>
              <audio controls>
                <source src="/audio/modelDiff/tibePlay.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">MyShell</h3>
              <audio controls>
                <source src="/audio/modelDiff/tibeMyShell.mp3" type="audio/mp3" />
                Your browser does not support the audio element.
              </audio>
            </div>
            <div className="flex flex-col space-y-4 items-center">
              <h3 className="text-2xl font-semibold">Descript</h3>
              <audio controls>
                <source src="/audio/modelDiff/tibeDescript.wav" type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
