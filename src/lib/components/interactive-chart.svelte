<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";
  import { onDestroy, onMount } from "svelte";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let selectedMetric: string | null = $state(null);
  let { experiment }: { experiment: Experiment } = $props();

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

  function destroyChart() {
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
  }

  function createChart(label: string, x: number[], y: number[]) {
    destroyChart();
    if (!chartCanvas) return;

    try {
      chartInstance = new Chart(chartCanvas, {
        type: "line",
        data: {
          labels: x,
          datasets: [
            {
              label,
              data: y,
              borderColor: "#6B7280", // gray-500 for minimalist look
              backgroundColor: "#F3F4F6", // gray-100 for subtle fill
              borderWidth: 1.5,
              tension: 0.2,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            intersect: false,
            mode: "index",
          },
          plugins: {
            legend: {
              display: false, // Hide legend for cleaner look
            },
            tooltip: {
              backgroundColor: "#ffffff",
              titleColor: "#111827",
              bodyColor: "#374151",
              borderColor: "#E5E7EB",
              borderWidth: 1,
              padding: 8,
              displayColors: false,
            },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: "Step",
                font: {
                  size: 12,
                  weight: 400,
                },
                color: "#6B7280",
              },
              grid: {
                display: false,
              },
              ticks: {
                color: "#9CA3AF",
                font: {
                  size: 11,
                },
              },
            },
            y: {
              title: {
                display: true,
                text: label,
                font: {
                  size: 12,
                  weight: 400,
                },
                color: "#6B7280",
              },
              grid: {
                color: "#F3F4F6",
              },
              ticks: {
                color: "#9CA3AF",
                font: {
                  size: 11,
                },
              },
            },
          },
        },
      });
    } catch (error) {
      console.error("Failed to create chart:", error);
    }
  }

  async function setSelectedMetric(metric: string) {
    selectedMetric = metric;
    const metrics = (await loadMetrics()) as Metric[];
    const loss = Object.groupBy(metrics, ({ name }) => name);
    const chart_targets = loss[metric];

    if (chart_targets) {
      chart_targets.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
      const steps = chart_targets.map((l) => l.step ?? 0);
      const values = chart_targets.map((l) => l.value);
      createChart(metric, steps, values);
    }
  }
</script>

<div class="space-y-4">
  {#if experiment.availableMetrics}
    <div class="flex flex-wrap gap-2">
      {#each experiment.availableMetrics as metric}
        <button
          class="px-3 py-1.5 text-sm text-gray-600 bg-gray-50
                 hover:bg-gray-100 rounded-sm transition-colors
                 {selectedMetric === metric ? 'bg-gray-200 text-gray-800' : ''}"
          onclick={() => setSelectedMetric(metric)}
        >
          {metric}
        </button>
      {/each}
    </div>
  {/if}

  {#if selectedMetric}
    <div class="h-64 w-full">
      <canvas bind:this={chartCanvas}></canvas>
    </div>
  {/if}
</div>
