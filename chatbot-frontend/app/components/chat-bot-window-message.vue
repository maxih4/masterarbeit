<script setup lang="ts">
const props = defineProps<{
    role: 'bot' | 'user',
    content: string,
    timestamp: number,
}>()
</script>

<template>

    <div class="mb-3">
        <div v-if="props.role === 'bot'">

            <div class='flex flex-row items-center gap-2'>
                <div>
                    <Avatar icon="pi pi-user" style="background-color: #dee9fc; color: #1a2551" shape="circle" />
                </div>
                <div>
                    <div class="text-xs text-gray-300 mb-1 flex justify-start">
                        Bot
                    </div>
                    <div class="bg-blue-100 rounded-xl p-3 text-black grow" style="word-wrap: break-word;">
                        {{ props.content }}
                    </div>
                    <div class="text-xs text-gray-300 mt-1 flex justify-start"> {{ new
                        Date(props.timestamp).toLocaleTimeString() }}
                    </div>
                </div>
            </div>
        </div>
        <div v-if="props.role === 'user'">
            <div class="text-xs text-gray-300 mb-1 flex  justify-end">
                User
            </div>
            <div class="rounded-xl p-3 text-black grow bg-gray-800 text-white" style="word-wrap: break-word;">
                {{ props.content }}
            </div>

            <div class="text-xs text-gray-300 mt-1 flex justify-end">
                {{ new Date(props.timestamp).toLocaleTimeString() }}
            </div>

        </div>
    </div>

</template>