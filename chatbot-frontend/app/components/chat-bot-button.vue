<script setup lang="ts">
const props = defineProps<{
    isOpen: boolean
}>()
</script>

<template>


    <Button v-if="props.isOpen" severity="secondary" type="button" icon="pi pi-angle-down" raised />
    <OverlayBadge v-else value="4" severity="danger">
        <Button type="button" severity="info" raised icon="pi pi-inbox" />
    </OverlayBadge>

</template>