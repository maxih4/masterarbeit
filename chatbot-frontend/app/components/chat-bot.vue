<script setup lang="ts">
import { useMessageStore } from '~~/stores/messageStore'


const store = useMessageStore()

const {  data, send,  } = useWebSocket(`ws://localhost:3000/api/websocket`)
watch(data, (newData) => {
  console.log('Received data:', newData)
  store.sendBotMessage(newData)
})
const chatBotIsOpen = ref(false)
const toggleChatBot = () => {
  chatBotIsOpen.value = !chatBotIsOpen.value
}

const sendMessage = (message: string) => {
  store.sendUserMessage(message)
  send(message)
}
</script>


<template>
  <div class="fixed bottom-5 right-5 flex flex-col items-end gap-2 px-4">

    <chat-bot-window v-if="chatBotIsOpen" @close-window="toggleChatBot" @send-message="sendMessage" />

    <chat-bot-button :is-open="chatBotIsOpen" @click="toggleChatBot" />
  </div>
</template>