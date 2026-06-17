
// uploadService.js
import axios from 'axios';

const baseURL = '/api'; //TODO 后端端口
const request = axios.create({baseURL})




export const query = (frame) => {
    return request.get('/process', {
        params: frame,
        responseType: 'arraybuffer' // 确保获取二进制数据
    })
};