<script lang="ts">
  import Chart from "chart.js/auto";
  import { onDestroy, onMount } from "svelte";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement;

  function destroyChart() {
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
  }

  function createChart() {
    destroyChart();
    if (chartCanvas) {
      try {
        chartInstance = new Chart(chartCanvas, {
          type: "line",
          data: {
            labels: [0, 1, 2],
            datasets: [
              {
                label: "loss",
                data: [0.5, 0.2, 0.1],
                borderColor: "rgb(75, 192, 192)",
                tension: 0.1,
              },
            ],
          },
          options: {
            responsive: true,
          },
        });
      } catch (error) {
        console.error("Failed to create chart:", error);
      }
    }
  }

  onMount(() => {
    createChart();
  });

  onDestroy(() => {
    destroyChart();
  });

  export function destroy() {
    destroyChart();
  }
</script>

<canvas bind:this={chartCanvas} id="myChart"></canvas>
