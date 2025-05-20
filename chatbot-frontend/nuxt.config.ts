// https://nuxt.com/docs/api/configuration/nuxt-config

import { definePreset } from '@primeuix/themes';
import Aura from '@primeuix/themes/aura';

const MyPreset = definePreset(Aura,{

});


export default defineNuxtConfig({
  compatibilityDate: '2025-05-15',
  devtools: { enabled: true },
  modules: [
    '@nuxt/eslint',
    '@primevue/nuxt-module',
    '@nuxtjs/tailwindcss',
    '@vueuse/nuxt',
    '@formkit/auto-animate/nuxt',
    '@pinia/nuxt'
  ],
  future: {
    compatibilityVersion: 4,
  },
   primevue: {
        options: {
            theme: {
                preset: MyPreset,
            }
        }
    },
      css: ['~/assets/global.css'],
      nitro:{
        experimental:{
          websocket:true
        }
      }
})