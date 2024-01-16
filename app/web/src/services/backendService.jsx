export class BackendService {
    constructor() {
        this.baseUrl = "http://localhost:8000";
    }

    async getModels() {
        const response = await fetch(this.baseUrl + "/get/models", {
            method: "GET",
            headers: {},
        });
        return response.json();
    }

    async selectModel(model) {
        const response = await fetch(this.baseUrl + "/select/model/"+model, {
            method: "get",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.json();
    }

    async selectSpeaker(speaker) {
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
            headers: {},
        });
        return response.json();
    }

    async getSpeakers() {
        const response = await fetch(this.baseUrl + "/get/model/speakers", {
            method: "GET",
            headers: {},
        });
        return response.json();
    }

    async getSelectedSpeaker() {
        const response = await fetch(this.baseUrl + "/get/selected/speaker", {
            method: "GET",
            headers: {},
        });
        return response.json();
    }
}