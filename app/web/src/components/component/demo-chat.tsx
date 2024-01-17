/**
 * This code was generated by v0 by Vercel.
 * @see https://v0.dev/t/gDV6BZbnwRt
 */
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuItem, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { AvatarImage, AvatarFallback, Avatar } from "@/components/ui/avatar"
import { Input } from "@/components/ui/input"
import { useEffect ,useState, useRef } from "react"
import { Navbar } from "@/components/component/navbar"
import { BackendService } from "@/services/backendService"

declare global {
  interface Window {
    webkitSpeechRecognition:any;
  }
}

export function DemoChat() {

  const recognitionRef = useRef<any>(null);
  const [transcript, setTranscript] = useState<string>("");

  const [language, setLanguage] = useState("English")
  const [languageItems, setLanguageItems] = useState([{
    key: 1,
    value: "English",
    short: "en"
  }, {
    key: 2,
    value: "Dutch",
    short: "nl"
  }
])


  const backendService = new BackendService()
  // const [backendService, setBackendService] = useState(new BackendService())

  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob>(new Blob());
  // const mediaRecorder = useRef<MediaRecorder | null>(null);


  const [audioData, setAudioData] = useState(null);
  const [loading, setLoading] = useState(false)
  const [modelItems, setModelItems] = useState([])
  const [model, setModel] = useState("No model selected")
  const [speaker, setSpeaker] = useState("Select a model first")
  const [speakerItems, setSpeakerItems] = useState([{
    key: 1,
    value: "Choose a model first"
  }])
  const [messages,setMessages] =useState(
    [{
    key: 1,
    value: "Hello, how can I assist you today?",
    speaker: "Openai"
  }]
  )

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

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    }
  }, [])

  const playWavFile = (base64Audio:any) => {
    const decodedAudio = atob(base64Audio);
    const audioArray = new Uint8Array(decodedAudio.length);
    for (let i = 0; i < decodedAudio.length; i++) {
        audioArray[i] = decodedAudio.charCodeAt(i);
    }

    const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    console.log(audioUrl)

    const audio = new Audio(audioUrl);

    // audio.play();
  }
  
  const addMessage = (message: any, speaker: string) => {
    if (speaker !== "User" && speaker !== "Openai") {
      console.error("Invalid speaker value. It must be 'User' or 'Openai'.");
      return; // or throw an error, depending on your desired behavior
    }

    setMessages((prevMessages) => [
      ...prevMessages,
      { key: prevMessages.length + 1, value: message, speaker: speaker },
    ]);
  };

  useEffect(() => {
    // check if the new message is from the user
    if (messages[messages.length - 1].speaker === "User") {
        backendService.getChatbotResponse(messages).then((res) => {
          // console.log(res)
          addMessage(res["text"],"Openai")
          setAudioData(res["audio"])
          playWavFile(res["audio"])
          setLoading(false)
        })
    }
  }, [messages]);

  const sendMessage = async () => {
    if (loading) {
      return;
    }
    if(speaker === "Select a model first" || speaker === "Wait for model to load" || speaker === "Loading speaker..." || speaker === "Choose speaker") {
      return;
    }

    const input = document.querySelector('input') as HTMLInputElement;
    // check if the input is empty or whitespace or shorter than 5 characters
    if (!input.value || !input.value.trim() || input.value.length < 5) {
      console.log("empty input")
      return;
    }
    const inputValue = input.value
    input.value = ""
    setLoading(true)
    addMessage(inputValue,"User")
    // set loading to false in useEffect of messages
  }

  const checkKey = (e:any) => {
    if (e.key === 'Enter') {
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
    setSpeaker(speaker)
  }

  const startRecording = () => {
    recognitionRef.current = new window.webkitSpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    if (language === "Dutch") {
      // get the short name of the language out of the languageItems array
      const shortName = languageItems.filter((item) => item.value === "Dutch")[0].short
      recognitionRef.current.lang = shortName
    } else {
      const shortName = languageItems.filter((item) => item.value === "English")[0].short
      recognitionRef.current.lang = shortName
    }
    console.log(recognitionRef.current.lang)

    const input = document.querySelector('input') as HTMLInputElement;
    recognitionRef.current.onresult = (event:any) => {
      const {transcript} = event.results[event.results.length - 1][0];
      setTranscript(transcript);
      input.value = transcript

    }
    setRecording(true);
    recognitionRef.current.start();
    // navigator.mediaDevices.getUserMedia({ audio: true })
    //   .then((stream) => {
    //     console
    //     mediaRecorder.current = new MediaRecorder(stream);
    //     const chunks:any = [];

    //     mediaRecorder.current.ondataavailable = (e:any) => {
    //       if (e.data.size > 0) {
    //         chunks.push(e.data);
    //       }
    //     };

    //     mediaRecorder.current.onstop = () => {
    //       const blob:any = new Blob(chunks, { type: 'audio/wav' });
    //       setAudioBlob(blob);
    //       const audioUrl = URL.createObjectURL(blob);
    //       console.log(audioUrl)
    //       const audio = new Audio(audioUrl);
    //       audio.play();

    //       const audioData = new FormData();
    //       audioData.append('audio', blob, 'audio.wav');
    //       backendService.sendAudioFile(audioData).then((res:any) => {
    //         console.log(res)
    //         // addMessage(res["text"],"Openai")
    //         // setAudioData(res["audio"])
    //         // playWavFile(res["audio"])
    //         // setLoading(false)
    //       } )


    //     };

    //     mediaRecorder.current.start();
    //     setRecording(true);
    //   })
    //   .catch((error) => {
    //     console.error('Error accessing microphone:', error);
    //   });
  };

  const stopRecording = () => {
    setRecording(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      console.log(transcript)
    }
    // if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
    //   mediaRecorder.current.stop();
    //   setRecording(false);
    // }
  };

  return (
    <div key="1" className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <Navbar />
      <div className="flex h-full overflow-y-auto">
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
              model === "No model selected" ? (
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
          <h2 className="text-lg font-semibold mt-8 mb-4">Language</h2>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">{language}</Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
              {languageItems.map((item:any) => (
                <DropdownMenuItem key={item.key} onClick={() => setLanguage(item.value)}>
                  <p className="block py-2 px-3 rounded" >
                    {item.value}
                  </p>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <h2 className="text-lg font-semibold mt-8 mb-4">Audio Files</h2>
          <div className="w-full space-y-4">
            { audioData && <audio className="w-full" controls>
              <source src={`data:audio/wav;rate=24000;base64,${audioData}`} type="audio/wav" />
              {/* <source src={`data:audio/wav;rate=24000,${audioData}`} type="audio/wav" /> */}
              Your browser does not support the audio element.
            </audio>}
          </div>
        </aside>
        <main className="flex-1 p-6 overflow-y-auto">
          <div className="flex flex-col h-full">
            <div className="flex-1 overflow-y-auto" >
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
                        <AvatarImage alt="Speaker 2" src="/user.jpg" />
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
                {/* <Button variant="dark" className="ml-2 mr-4" type="button">
                  <img alt="Microphone" className="h-6 w-6" src="/mic.svg" />
                  <span className="sr-only">Record voice</span>
                </Button> */}
                <Button variant="dark" className={`ml-2 mr-4`}
                  onClick={recording ? stopRecording : startRecording}
                  type="button"
                >
                  {recording ? 'Stop Recording' : 'Record Voice'}
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
