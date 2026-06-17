import { createRouter, createWebHistory } from 'vue-router'
//导入组件
import Home from '@/views/home.vue'
// import TestPage from '@/views/test_page.vue'

//定义路由关系
const routes = [
    { path: '/', component: Home },
    // {path:'/test_page',component: TestPage},
]


//创建路由器
const router = createRouter({
    history: createWebHistory(),
    routes: routes
})

//导出路由
export default router
