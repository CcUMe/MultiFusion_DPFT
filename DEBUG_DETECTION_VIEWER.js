// ===== 检测框显示问题诊断脚本 =====
// 在浏览器开发者工具 Console 中复制粘贴以下代码

// 1. 基础检查
console.log('=== 基础Canvas检查 ===');
const canvas = document.querySelector('.detection-canvas');
if (!canvas) {
  console.error('❌ Canvas元素不存在！');
} else {
  console.log('✅ Canvas元素存在');
  console.log('  Canvas.width:', canvas.width);
  console.log('  Canvas.height:', canvas.height);
  console.log('  Canvas.style.width:', canvas.style.width);
  console.log('  Canvas.style.height:', canvas.style.height);
  console.log('  Canvas.clientWidth:', canvas.clientWidth);
  console.log('  Canvas.clientHeight:', canvas.clientHeight);
  
  const rect = canvas.getBoundingClientRect();
  console.log('  Canvas位置:', {
    top: rect.top,
    left: rect.left,
    width: rect.width,
    height: rect.height
  });
  
  // 检查是否可见
  if (rect.width === 0 || rect.height === 0) {
    console.error('❌ Canvas尺寸为0，不可见！');
  } else {
    console.log('✅ Canvas尺寸正常');
  }
  
  // 检查z-index
  const zindex = window.getComputedStyle(canvas).zIndex;
  console.log('  Canvas z-index:', zindex);
  
  // 检查display
  const display = window.getComputedStyle(canvas).display;
  console.log('  Canvas display:', display);
  
  // 检查position
  const position = window.getComputedStyle(canvas).position;
  console.log('  Canvas position:', position);
}

// 2. 检查图像
console.log('\n=== 图像检查 ===');
const img = document.querySelector('.base-image');
if (!img) {
  console.warn('⚠️  图像元素不存在');
} else {
  console.log('✅ 图像元素存在');
  console.log('  图像src:', (img as HTMLImageElement).src?.substring(0, 100));
  console.log('  图像已加载:', (img as HTMLImageElement).complete);
  console.log('  图像自然尺寸:', {
    width: (img as HTMLImageElement).naturalWidth,
    height: (img as HTMLImageElement).naturalHeight
  });
  console.log('  图像显示尺寸:', {
    width: (img as HTMLImageElement).clientWidth,
    height: (img as HTMLImageElement).clientHeight
  });
}

// 3. 检查容器
console.log('\n=== 容器检查 ===');
const imageSection = canvas?.parentElement;
if (imageSection) {
  console.log('✅ 容器存在');
  console.log('  容器尺寸:', {
    width: imageSection.clientWidth,
    height: imageSection.clientHeight
  });
  
  const containerStyle = window.getComputedStyle(imageSection);
  console.log('  容器position:', containerStyle.position);
  console.log('  容器display:', containerStyle.display);
  console.log('  容器overflow:', containerStyle.overflow);
}

// 4. 检查Canvas上下文
console.log('\n=== Canvas上下文检查 ===');
if (canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.error('❌ 无法获取2D上下文！');
  } else {
    console.log('✅ Canvas 2D上下文存在');
    
    // 尝试绘制测试内容
    console.log('\n📝 绘制测试...');
    ctx.fillStyle = 'red';
    ctx.fillRect(10, 10, 100, 100);
    console.log('✅ 已绘制红色测试矩形 (10,10,100,100)');
    console.log('   如果看不到红色方块，Canvas可能被CSS隐藏或z-index问题');
  }
}

// 5. 使用全局调试函数
console.log('\n=== 使用全局调试函数 ===');
if ((window as any).debugDetectionViewer) {
  console.log('✅ 全局调试函数已可用');
  console.log('调用以下函数获取更多信息:');
  console.log('  - debugDetectionViewer.canvas() - 显示Canvas状态');
  console.log('  - debugDetectionViewer.detections() - 显示检测框数据');
  console.log('  - debugDetectionViewer.images() - 显示图像状态');
  console.log('  - debugDetectionViewer.testDraw() - 在Canvas上绘制测试红色方块');
  console.log('  - debugDetectionViewer.redraw() - 手动重绘');
} else {
  console.warn('⚠️  全局调试函数未注册，请确保DetectionViewer已挂载');
}

// 6. 监控数据流
console.log('\n=== 数据流监控 ===');
console.log('查看浏览器Console中的以下日志：');
console.log('  - 【ControlPanel】 开头：数据来自ControlPanel');
console.log('  - 【解析检测数据】 开头：数据解析过程');
console.log('  - 【Canvas绘制】 开头：Canvas绘制过程');
console.log('  - 【投影】 开头：坐标投影过程');
console.log('\n💡 建议：打开Console并按照类别过滤日志，追踪完整的数据流');

// 7. 总结
console.log('\n=== 诊断总结 ===');
const issues = [];
if (!canvas) issues.push('❌ Canvas元素不存在');
if (canvas && (canvas.width === 0 || canvas.height === 0)) issues.push('❌ Canvas尺寸为0');
if (canvas && canvas.clientWidth === 0) issues.push('❌ Canvas CSS宽度为0');
if (!img) issues.push('⚠️  图像未加载');
if (img && !(img as HTMLImageElement).complete) issues.push('⚠️  图像未完全加载');

if (issues.length === 0) {
  console.log('✅ 所有基础检查通过！');
  console.log('如果检测框仍未显示，请：');
  console.log('  1. 确认后端正在发送检测框数据');
  console.log('  2. 检查【ControlPanel】日志中的数据格式');
  console.log('  3. 检查【解析检测数据】日志中的解析结果');
  console.log('  4. 检查【Canvas绘制】日志中的绘制坐标是否超出范围');
} else {
  console.log('发现问题：');
  issues.forEach(issue => console.log('  ' + issue));
}
