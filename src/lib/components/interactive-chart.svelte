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
              borderColor: "#6B7280",
              backgroundColor: "#F3F4F6",
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
              titleColor: "#1e293b",
              bodyColor: "#475569",
              borderColor: "#e2e8f0",
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
                display: true,
                color: "#f1f5f9",
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
                display: true,
                color: "#f1f5f9",
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

<div
  class="rounded-md border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200"
>
  <div class="p-4 space-y-4">
    {#if experiment.availableMetrics}
      <div class="flex flex-wrap gap-2">
        {#each experiment.availableMetrics as metric}
          <button
            class="px-3 py-1.5 text-sm font-medium
                   rounded-md border border-gray-200
                   transition-all duration-150 ease-in-out
                   hover:border-gray-300 hover:bg-gray-50 hover:shadow-sm
                   focus:outline-none focus:ring-2 focus:ring-blue-500/20
                   {selectedMetric === metric
              ? 'bg-red-50 text-gray-600 border-red-200 shadow-sm'
              : 'bg-white text-gray-600'}"
            onclick={() => setSelectedMetric(metric)}
          >
            {metric}
          </button>
        {/each}
      </div>
    {/if}

    {#if selectedMetric}
      <div
        class="relative h-64 w-full rounded-sm border border-gray-200 bg-gray-50/50 overflow-hidden"
      >
        <div class="absolute inset-0 bg-white m-[1px] rounded-sm">
          <canvas bind:this={chartCanvas} class="p-2"></canvas>
        </div>
      </div>

      <!-- <div class="mt-2 text-xs text-gray-400 text-center">
        Click and drag to zoom • Double click to reset
      </div> -->
    {/if}
  </div>
</div>

<style>
  canvas {
    background-image: radial-gradient(#f1f5f9 1px, transparent 1px);
    background-size: 20px 20px;
    background-position: -10px -10px;
  }
</style>
