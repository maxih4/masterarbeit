import { defineStore } from 'pinia'
import type { ChatMessage } from '~/utils/types'


export const useMessageStore = defineStore('message', {

    state: () => ({
        messages: [] as ChatMessage[],
    }),
    actions: {
        sendUserMessage(message: string) {
            this.messages.push({
                role: 'user',
                content: message,
                timestamp: Date.now(),
            })
        },
        sendBotMessage(message: string) {
            this.messages.push({
                role: 'bot',
                content: message,
                timestamp: Date.now(),
            })
        },
    },
    getters:{
        getMessages:(state) => {
            return state.messages.sort((a, b) => {
                return a.timestamp - b.timestamp
            })
        }
    }
})