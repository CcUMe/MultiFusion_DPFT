<template>
  <!--  <div id="camera-settings"></div>     Add a div to display camera settings -->

  <div id="plotly-3d" style="width: 100%; height: 100%;"></div>

</template>

<script setup>
import {onMounted, watch, ref} from 'vue';
import Plotly from 'plotly.js-dist';

const props = defineProps({
  trajectories: {
    type: Array,
    required: true
  }
});

// 新增：保存当前相机参数
const currentCamera = ref({
  "center": {
    "x": -0.0274429048984906,
    "y": 0.26747533595106715,
    "z": -0.1593106339324853
  },
  "eye": {
    "x": 2.1235356716053766,
    "y": 0.7018283144423973,
    "z": 0.7718852411958463
  },
});

const updatePlot = () => {
  const data = props.trajectories.map((trajectory, index) => {
    const x = trajectory.points.map(p => p.x);
    const y = trajectory.points.map(p => p.y);
    const z = trajectory.points.map(p => p.z);

    // 找到最后一个非null的点
    let lastValidIdx = x.length - 1;
    while (lastValidIdx >= 0 && (x[lastValidIdx] == null || y[lastValidIdx] == null || z[lastValidIdx] == null)) {
      lastValidIdx--;
    }
    // 新增：构造textArr，除最后一个点外显示置信度
    let textArr = new Array(x.length).fill('');
    for (let i = 0; i < x.length; i++) {
      if (i === lastValidIdx) {
        textArr[i] = `${trajectory.name}：${trajectory.class}<br>准确率：100%<br>虚警率：0`;
      } else if (trajectory.points[i] && trajectory.points[i].confidence !== undefined && trajectory.points[i].confidence !== null) {
        textArr[i] = `${(trajectory.points[i].confidence).toFixed(2)}`; //#TODO 添加一个按钮，显示和隐藏置信度？或者鼠标停留在上面才显示置信度？
      }
    }

    return {
      x: x,
      y: y,
      z: z,
      mode: 'lines+markers+text',
      type: 'scatter3d',
      line: {
        color: trajectory.color || 'blue',
        width: 2
      },
      marker: {
        size: 0.8
      },
      name: trajectory.name + ': ' + trajectory.class,
      text: textArr,
      textposition: 'top center',
      textfont: {
        color: trajectory.color || 'blue',
        size: 10
      },
      // 修改：显示legend内容
      hovertemplate: '置信度：%{text}<extra>%{fullData.name}</extra>'
    };
  });

  const layout = {
    autosize: true,
    scene: {
      xaxis: {
        title: {text:''},
        showticklabels: false,
      },
      yaxis: {
        title: {text:''},
        showticklabels: false,
      },
      zaxis: {
        title: {text:''},
        showticklabels: false,
      },
      aspectmode: 'manual',
      aspectratio: {x: 2 * 0.65, y: 3.5 * 0.65, z: 1.6 * 0.65},
      // 使用当前相机参数
      camera: currentCamera.value
    },
    margin: {
      l: 0,
      r: 0,
      b: 0,
      t: 0,
      pad: 0
    },
    modebar: {
      // remove: [
      //   'zoom', 'Pan', 'plotly-logomark', 'zoomIn2d', 'zoomOut2d', 'pan2d', 'select2d', 'lasso2d', 'orbitRotation', 'tableRotation',
      //   'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'hoverClosest3d',
      //   'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toImage', 'sendDataToCloud',
      //   'editInChartStudio', 'zoom3d', 'pan3d', 'toggleSpikelines', 'autoScale2d', 'resetScale2d'
      // ]
      // remove: ['zoom', 'Pan', 'plotly-logomark', 'zoomIn2d', 'zoomOut2d', 'pan2d', 'select2d', 'lasso2d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews']
    },
    displaylogo: false,
    legend: {
      x: 1, // Position the legend outside the plot area
      y: 0.955, // Position the legend outside the plot area
      xanchor: 'right', // Anchor the legend to the right
      yanchor: 'top', // Anchor the legend to the top
      orientation: 'v', // Display the legend horizontally
      bgcolor: 'rgba(210,88,88,0)', // Background color
      bordercolor: 'rgb(171,171,171)', // Border color
      borderwidth: 1, // Border width
      visible: true, // 明确设置legend一直可见
      // 可选：禁用点击隐藏轨迹
      // itemclick: false,
      // itemdoubleclick: false,
      font: {
        // family: '微软雅黑', // Font family
         //size: 12, // Font size
        // color: 'black' // Font color
      }
    }
  };

  Plotly.react('plotly-3d', data, layout);
};

// 监听相机参数变化的，显示在div中
// onMounted(() => {
//   updatePlot();
//   const plotDiv = document.getElementById('plotly-3d');
//   plotDiv.on('plotly_relayout', (eventData) => {
//     if (eventData['scene.camera']) {
//       currentCamera.value = eventData['scene.camera'];
//       // 显示相机参数
//       document.getElementById('camera-settings').innerText =
//         JSON.stringify(eventData['scene.camera'], null, 2);
//     }
//   });
// });


// 这段代码是 Vue 的侦听器（watch），作用是监听 props.trajectories 的变化，
// 只要 trajectories（及其内部任意属性）发生变化，就会自动调用 updatePlot() 方法，
// 重新绘制 3D 轨迹图。
// {deep: true} 表示深度监听，能捕捉到对象或数组内部的变化。
watch(() => props.trajectories, () => {
  updatePlot();
}, {deep: true});
</script>

<style scoped>
/* Styles can be customized as needed */
</style>