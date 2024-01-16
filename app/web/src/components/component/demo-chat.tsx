/**
 * This code was generated by v0 by Vercel.
 * @see https://v0.dev/t/gDV6BZbnwRt
 */
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuItem, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { AvatarImage, AvatarFallback, Avatar } from "@/components/ui/avatar"
import { Input } from "@/components/ui/input"
import { useEffect ,useState } from "react"
import { Navbar } from "@/components/component/navbar"
import { BackendService } from "@/services/backendService"

export function DemoChat() {
  const backendService = new BackendService()
  
  const [modelItems, setModelItems] = useState([])
  const [model, setModel] = useState("No model selected")
  const [speaker, setSpeaker] = useState("Select a model first")
  const [speakerItems, setSpeakerItems] = useState([{
    key: 1,
    value: "Speaker 1"
  },{
    key: 2,
    value: "Speaker 2"
  },{
    key: 3,
    value: "Speaker 3"
  }])

  const checkSpeaker = async (model:any) => {
    setSpeaker("Loading speakers...")
    let speakers:any = backendService.getSpeakers()
    let selectedSpeaker:any = backendService.getSelectedSpeaker()
    speakers = await speakers
    selectedSpeaker = await selectedSpeaker
    setSpeakerItems(speakers.map((speaker:any, index:any) => ({ key: index + 1, value: speaker })));
    if (selectedSpeaker["selected"] === "None") {
      setSpeaker("Choose speaker")
    } else {
      // console.log(selectedSpeaker)
      setSpeaker(selectedSpeaker["selected"])
    }
    // setSpeaker(speakers[0])
  }

  useEffect(() => {
    // check if the backend has a model selected
    backendService.getSelectedModel().then((res) => {
      if (res["selected"] === "None") {
        setModel("No model selected")
      } else {
        setModel(res["selected"])
        checkSpeaker(res["selected"])
      }
    })
    // get the models for the dropdown
    backendService.getModels().then((res) => {
      // console.log(res)
      setModelItems(res.map((model:any, index:any) => ({ key: index + 1, value: model })));
    })
  }, [])


  const [messages,setMessages] =useState(
    [{
    key: 1,
    value: "Hello, how can I assist you today?",
    speaker: "Openai"
  },{
    key: 2,
    value: "I need help with my order.",
    speaker: "User"
  }]
  )
  

  const sendMessage = () => {
    console.log("send button pressed")
  }
  const checkKey = (e:any) => {
    if (e.key === 'Enter') {
      console.log("enter pressed")
      sendMessage()
    }
  }

  const selectedModel = async (model:any) => {
    setModel("Loading model...")
    setSpeaker("Wait for model to load")
    const models = await backendService.selectModel(model)
    console.log(models)
    setModel(model)
    checkSpeaker(model)
    // setSpeaker("Choose speaker")
  }

  const selectedSpeaker = async (speaker:any) => {
    setSpeaker("Loading speaker...")
    const selectedSpeaker = await backendService.selectSpeaker(speaker)
    console.log(selectedSpeaker)
    setSpeaker(speaker)
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
              {modelItems.map((item:any) => (
                <DropdownMenuItem key={item.key} onClick={() => selectedModel(item.value)}>
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
                <DropdownMenuItem key={1}>
                  <p className="block py-2 px-3 rounded">
                    Select a model first
                  </p>
                </DropdownMenuItem>
              ) : (
                speakerItems.map((item) => (
                  <DropdownMenuItem key={item.key} onClick={() => selectedSpeaker(item.value)}>
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
                {messages.map((item) => (
                  (item.speaker === "Openai") ?(
                    <div className="flex items-end" key={item.key}>
                      <Avatar className="w-10 h-10 mr-4">
                        <AvatarImage alt="Speaker 1" src="/openai.png" />
                        <AvatarFallback>S1</AvatarFallback>
                      </Avatar>
                      <div className="bg-gray-200 dark:bg-gray-800 rounded-lg px-4 py-2 max-w-1/2 max-w-[650px]">
                        <p>{item.value}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-end justify-end" key={item.key}>
                      <div className="bg-blue-500 text-white rounded-lg px-4 py-2 mr-4 max-w-1/2 max-w-[650px]">
                        <p>{item.value}</p>
                      </div>
                      <Avatar className="w-10 h-10">
                        <AvatarImage alt="Speaker 2" src="/user.png" />
                        <AvatarFallback>S2</AvatarFallback>
                      </Avatar>
                    </div>
                  )
                  ))}
              </div>
            </div>
            <div className="mt-4 border-t dark:border-gray-700 pt-4">
              <div className="flex">
                <Input className="flex-1 mr-2" placeholder="Type your message..." onKeyPress={checkKey}/>
                <Button variant="dark" className="ml-2 mr-4" type="button">
                  <img alt="Microphone" className="h-6 w-6" src="/mic.svg" />
                  <span className="sr-only">Record voice</span>
                </Button>
                <Button variant="dark" onClick={sendMessage}>Send</Button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
