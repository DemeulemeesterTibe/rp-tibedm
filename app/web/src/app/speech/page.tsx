"use client";
import { Navbar } from "@/components/component/navbar"
import { SpeechSynthesize } from "@/components/component/speech-synthesize"
export default function Speech() {
    return (
        <div>
            <Navbar />
            <SpeechSynthesize />
            <h1>Speech</h1>
        </div>
    )
  }