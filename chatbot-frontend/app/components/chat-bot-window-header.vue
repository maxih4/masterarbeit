<script setup lang="ts">
defineEmits<{
  (e: 'closeWindow'): void
}>()
</script>

<template>
    <div class="h-1/7 py-4 px-3 chatbotheader flex flex-row justify-between pr-6 rounded-t-xl items-center">
        <Image src="logo.png" width="100" />
        <h3 class="text-xl font-mono">Copilot</h3>
<Button icon ="pi pi-times"  severity="secondary" variant="text" rounded @click="$emit('closeWindow')"/>
    </div>
</template>

<style>
.chatbotheader {
    background: #009FE3;
    background: radial-gradient(circle, rgba(0, 159, 227, 1) 0%, rgba(0, 125, 186, 1) 30%, rgba(0, 48, 93, 1) 100%);
}
</style>