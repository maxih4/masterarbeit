<script setup lang="ts">
import { useMessageStore } from '~~/stores/messageStore';



const { height } = useWindowSize({initialHeight:200})

const heightValue= computed(()=> Math.max(height.value*0.5, 200))
const emit = defineEmits<{
  (e: 'closeWindow'): void,
    (e: 'sendMessage', message: string): void

}>()

const store = useMessageStore()

const scrollContent = ref<HTMLElement | null>(null)
const sendMessage = (message: string) => {
    emit('sendMessage', message)

}
</script>

<template>
          

<div class=" w-96 rounded-b-xl overflow-hidden">
    <!-- Fixed Header -->
    <div class="sticky top-0 z-10">
        <ChatBotWindowHeader @close-window="$emit('closeWindow')"/>
    </div>

    <!-- Scrollable Content -->
    <div ref="scrollContent" class="overflow-y-auto p-4 border-solid border border-white"  v-auto-animate :style="{ height: heightValue + 'px' }">
        
            <ChatBotWindowMessage v-for="(message, index) in store.getMessages" :key="index" :role="message.role" :content="message.content" :timestamp="message.timestamp" />

            
    </div>
       <div class="sticky bottom-0 z-10">
        <ChatBotWindowInput @send-message="sendMessage"/>
    </div>
</div>


</template>