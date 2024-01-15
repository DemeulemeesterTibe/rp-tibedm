/**
 * This code was generated by v0 by Vercel.
 * @see https://v0.dev/t/gDV6BZbnwRt
 */
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuItem, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { AvatarImage, AvatarFallback, Avatar } from "@/components/ui/avatar"
import { Input } from "@/components/ui/input"
import { useState } from "react"
import { Navbar } from "@/components/component/navbar"

export function DemoChat() {

  const modelItems = [{
    key: 1,
    value: "Model 1"
  },{
    key: 2,
    value: "Model 2"
  },{
    key: 3,
    value: "Model 3"
  }]

  const speakerItems = [{
    key: 1,
    value: "Speaker 1"
  },{
    key: 2,
    value: "Speaker 2"
  },{
    key: 3,
    value: "Speaker 3"
  }]
  
  const [model, setModel] = useState("Choose model")
  const [speaker, setSpeaker] = useState("Select a model first")

  const sendButton = () => {
    console.log("send button pressed")
  }

  return (
    <div key="1" className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <Navbar />
      <div className="flex h-full">
        <aside className="w-64 bg-white dark:bg-gray-800 border-r dark:border-gray-700 p-4">
          <h2 className="text-lg font-semibold mb-4">Models</h2>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">{model}</Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
              {modelItems.map((item) => (
                <DropdownMenuItem key={item.key} onClick={() => setModel(item.value)}>
                  <p className="block py-2 px-3 rounded" >
                    {item.value}
                  </p>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <h2 className="text-lg font-semibold mt-8 mb-4">Speakers</h2>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">{speaker}</Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
            {
              model === "Choose model" ? (
                <DropdownMenuItem>
                  <p className="block py-2 px-3 rounded">
                    Select a model first
                  </p>
                </DropdownMenuItem>
              ) : (
                speakerItems.map((item) => (
                  <DropdownMenuItem key={item.key} onClick={() => setSpeaker(item.value)}>
                    <p className="block py-2 px-3 rounded">{item.value}</p>
                  </DropdownMenuItem>
                ))
              )
            }
            </DropdownMenuContent>
          </DropdownMenu>
        </aside>
        <main className="flex-1 p-6">
          <div className="flex flex-col h-full">
            <div className="flex-1 overflow-y-auto">
              <div className="flex flex-col space-y-4">
                <div className="flex items-end">
                  <Avatar className="w-10 h-10 mr-4">
                    <AvatarImage alt="Speaker 1" src="/openai.png" />
                    <AvatarFallback>S1</AvatarFallback>
                  </Avatar>
                  <div className="bg-gray-200 dark:bg-gray-800 rounded-lg px-4 py-2">
                    <p>Hello, how can I assist you today?</p>
                  </div>
                </div>
                <div className="flex items-end justify-end">
                  <div className="bg-blue-500 text-white rounded-lg px-4 py-2 mr-4">
                    <p>I need help with my order.</p>
                  </div>
                  <Avatar className="w-10 h-10">
                    <AvatarImage alt="Speaker 2" src="/user.png" />
                    <AvatarFallback>S2</AvatarFallback>
                  </Avatar>
                </div>
              </div>
            </div>
            <div className="mt-4 border-t dark:border-gray-700 pt-4">
              <div className="flex">
                <Input className="flex-1 mr-2" placeholder="Type your message..." />
                <Button variant="dark" className="ml-2 mr-4" type="button">
                  <img alt="Microphone" className="h-6 w-6" src="/mic.svg" />
                  <span className="sr-only">Record voice</span>
                </Button>
                <Button variant="dark" onClick={sendButton}>Send</Button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
