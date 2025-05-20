<script setup lang="ts">

const messsageInput = ref('')

const emit = defineEmits<{
  (e: 'sendMessage', message: string): void
}>()

const sendMessage = () => {
emit('sendMessage', messsageInput.value)
  messsageInput.value = ''
}
</script>

<template><div class="bg-white flex flex-row items-center p-3 gap-3" @keypress.enter="sendMessage">
<Textarea v-model="messsageInput" class="rounded-xl" style="resize: none" rows="2" variant="filled" fluid type="text" size="normal" placeholder="Bitte Text eingeben"/>
<div><Button icon="pi pi-send" severity="secondary" rounded @click="sendMessage"/></div>
</div>

</template>