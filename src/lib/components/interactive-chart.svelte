<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let { experiment }: { experiment: Experiment } = $props();

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
              borderColor: "#60a5fa",
              backgroundColor: "rgba(96, 165, 250, 0.2)",
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
              backgroundColor: "#1f2937",
              titleColor: "#e5e7eb",
              bodyColor: "#d1d5db",
              borderColor: "#374151",
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
                color: "#9ca3af",
              },
              grid: {
                display: true,
                color: "rgba(75, 85, 99, 0.4)",
              },
              ticks: {
                color: "#d1d5db",
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
                color: "#9ca3af",
              },
              grid: {
                display: true,
                color: "rgba(75, 85, 99, 0.4)",
              },
              ticks: {
                color: "#d1d5db",
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

  let selectedMetric: string | null = $state(null);
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


<div class="p-4 space-y-4">
{#if experiment.availableMetrics}
    <div class="flex flex-wrap gap-2">
    {#each experiment.availableMetrics as metric}
        <button
        class="px-3 py-1.5 text-sm font-medium
                rounded-md border border-gray-700
                transition-all duration-150 ease-in-out
                hover:border-gray-600 hover:bg-gray-700 hover:shadow-xs
                focus:outline-hidden focus:ring-2 focus:ring-blue-500/20
                {selectedMetric === metric
            ? 'bg-blue-900/40 text-blue-300 border-blue-700 shadow-xs'
            : 'bg-gray-800 text-gray-300'}"
        onclick={() => setSelectedMetric(metric)}
        >
        {metric}
        </button>
    {/each}
    </div>
{/if}

{#if selectedMetric}
    <div
    class="relative h-64 w-full rounded-xs border border-gray-700 bg-gray-800/50 overflow-hidden"
    >
    <div class="absolute inset-0 bg-gray-800 m-[1px] rounded-xs">
        <canvas bind:this={chartCanvas} class="p-2"></canvas>
    </div>
    </div>
{/if}
</div>


<style>
  canvas {
    background-image: radial-gradient(rgba(75, 85, 99, 0.4) 1px, transparent 1px);
    background-size: 20px 20px;
    background-position: -10px -10px;
  }
</style>
