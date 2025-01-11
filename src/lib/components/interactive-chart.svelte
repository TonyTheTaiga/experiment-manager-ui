<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";
  import { onDestroy, onMount } from "svelte";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement;

  let { experiment }: { experiment: Experiment } = $props();

  onMount(async () => {
    const metrics = (await loadMetrics()) as Metric[];
    let metricName = "epoch_loss";
    const loss = Object.groupBy(metrics, ({ name }) => name);
    const losses = loss[metricName];

    if (losses) {
      losses.sort((a, b) => {
        const stepA = a.step ?? 0;
        const stepB = b.step ?? 0;
        return stepA - stepB;
      });
      const steps = losses.map((l) => l.step ?? 0);
      const values = losses.map((l) => l.value);
      createChart(metricName, steps, values);
    }
  });

  onDestroy(() => {
    destroyChart();
  });

  async function loadMetrics() {
    try {
      const response = await fetch(`/api/experiments/${experiment.id}/metrics`);
      if (!response.ok) {
        throw new Error(`Failed to load metrics: ${response.statusText}`);
      }
      return await response.json();
    } catch (e) {
      console.error("Error loading metrics:", e);
      return null;
    }
  }

  export function destroy() {
    destroyChart();
  }

  function destroyChart() {
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
  }

  function createChart(label: string, x: number[], y: number[]) {
    destroyChart();
    if (!chartCanvas) {
      console.error("Chart canvas not found");
      return;
    }

    try {
      chartInstance = new Chart(chartCanvas, {
        type: "line",
        data: {
          labels: x,
          datasets: [
            {
              label: label,
              data: y,
              borderColor: "rgb(75, 192, 192)",
              tension: 0.1,
            },
          ],
        },
        options: {
          responsive: true,
          scales: {
            x: {
              title: {
                display: true,
                text: "Step",
              },
            },
            y: {
              title: {
                display: true,
                text: label,
              },
            },
          },
        },
      });
    } catch (error) {
      console.error("Failed to create chart:", error);
    }
  }
</script>

<canvas bind:this={chartCanvas} id="myChart"></canvas>
