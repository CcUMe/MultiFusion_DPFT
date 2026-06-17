/**
 * WebSocket 数据服务
 * 处理实时数据接收和后端通信
 */

export interface DetectionData {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  confidence: number;
  id?: string;
}

export interface ProcessingResult {
  detections: DetectionData[];
  timestamp: number;
  processingTime: number;
  status: 'success' | 'error';
  message?: string;
}

class DataService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  private pingInterval: NodeJS.Timeout | null = null;
  private listeners: { [key: string]: Function[] } = {
    'connection-open': [],
    'connection-close': [],
    'connection-error': [],
    'data-received': [],
    'data-error': []
  };

  /**
   * 连接 WebSocket
   */
  connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log('[DataService] WebSocket 已连接');
          this.reconnectAttempts = 0;
          this.startPingInterval();
          this.emit('connection-open');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as ProcessingResult;
            this.emit('data-received', data);
          } catch (error) {
            console.error('[DataService] 数据解析错误:', error);
            this.emit('data-error', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('[DataService] WebSocket 错误:', error);
          this.emit('connection-error', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('[DataService] WebSocket 已断开');
          this.stopPingInterval();
          this.emit('connection-close');
          this.attemptReconnect(url);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * 断开连接
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stopPingInterval();
  }

  /**
   * 发送数据
   */
  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('[DataService] WebSocket 未连接，无法发送数据');
    }
  }

  /**
   * 注册事件监听器
   */
  on(event: string, callback: Function): void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  /**
   * 移除事件监听器
   */
  off(event: string, callback: Function): void {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  /**
   * 触发事件
   */
  private emit(event: string, ...args: any[]): void {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(...args));
    }
  }

  /**
   * 启动心跳检测
   */
  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // 每 30 秒发送一次
  }

  /**
   * 停止心跳检测
   */
  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * 尝试重新连接
   */
  private attemptReconnect(url: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      console.log(`[DataService] ${delay}ms 后尝试重新连接 (第 ${this.reconnectAttempts} 次)`);
      
      setTimeout(() => {
        this.connect(url).catch(() => {
          // 重新连接失败，继续尝试
        });
      }, delay);
    } else {
      console.error('[DataService] 达到最大重连次数');
    }
  }

  /**
   * 获取连接状态
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// 导出单例
export default new DataService();
