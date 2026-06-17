import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
// import vueDevTools from 'vite-plugin-vue-devtools' //TODO 在这里

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        vue(),
        // vueDevTools(),
    ],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        },
    },
    server: { //dev 模式下
        proxy: {
            '/api': { //获取路径中包含了/api的请求
                target: 'http://10.123.2.10:39125',//后台服务所在的源
                changeOrigin: true, //修改源
                rewrite: (path) => path.replace(/^\/api/, '')  //api替换为''
            }

        },
        host: '0.0.0.0',
        port: 5010
    },
    preview: { //preview模式下
        proxy: {
            '/api': { //获取路径中包含了/api的请求
                target: 'http://10.123.2.10:39125',//后台服务所在的源
                changeOrigin: true, //修改源
                rewrite: (path) => path.replace(/^\/api/, '')  //api替换为''
            }

        },
        host: '0.0.0.0',
        port: 5010 // 确保在预览模式下也使用5000端口
    }
})
