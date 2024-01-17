require('dotenv').config();

export class BackendService {
    baseUrl: string;
    constructor() {
        // console.log(process.env.BACKEND_URL)
        this.baseUrl = process.env.BACKEND_URL as string;
    }

    async getModels() {
        const response = await fetch(this.baseUrl + "/get/models", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async selectModel(model:string) {
        const response = await fetch(this.baseUrl + "/select/model/"+model, {
            method: "get",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async selectSpeaker(speaker:string) {
        const response = await fetch(this.baseUrl + "/select/model/speaker/"+speaker, {
            method: "get",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async getSelectedModel() {
        const response = await fetch(this.baseUrl + "/get/selected/model", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async getSpeakers() {
        const response = await fetch(this.baseUrl + "/get/model/speakers", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async getSelectedSpeaker() {
        const response = await fetch(this.baseUrl + "/get/selected/speaker", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async getChatbotResponse(messages:any) {
        const response = await fetch(this.baseUrl + "/run/openai/completion", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(messages),
        });
        return response.json();
    }

    async sendAudioFile(file:any) {
        const response = await fetch(this.baseUrl + "/get/text/from/audio", {
            method: "POST",
            body: file,
        });
        return response.json();
    }
}