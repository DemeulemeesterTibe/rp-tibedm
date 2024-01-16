// require('dotenv').config();
import OpenAI from "openai";

export class OpenAIService {
    constructor() {
        console.log(typeof this.api_key);
        this.openai = new OpenAI();
    }

    async getText(messages) {
        const completion = await openai.chat.completions.create({
            messages: [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}],
            model: "gpt-3.5-turbo",
        });
        return completion.choices[0];
    }
}