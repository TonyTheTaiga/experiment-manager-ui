<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";
  import { BarChart4 } from "lucide-svelte";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let { experiment }: { experiment: Experiment } = $props();
  let isLoading: boolean = $state(false);

  async function loadMetrics() {
    try {
      isLoading = true;
      const response = await fetch(`/api/experiments/${experiment.id}/metrics`);
      if (!response.ok) {
        throw new Error(`Failed to load metrics: ${response.statusText}`);
      }
      return await response.json();
    } catch (e) {
      console.error("Error loading metrics:", e);
      return null;
    } finally {
      isLoading = false;
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
              borderColor: "var(--color-ctp-blue)",
              backgroundColor: "rgba(137, 180, 250, 0.12)",
              borderWidth: 2,
              tension: 0.3,
              fill: true,
              pointBackgroundColor: "var(--color-ctp-lavender)",
              pointBorderColor: "var(--color-ctp-mantle)",
              pointRadius: 3,
              pointHoverRadius: 5,
              pointHoverBackgroundColor: "var(--color-ctp-mauve)",
              pointHoverBorderColor: "var(--color-ctp-base)",
              pointHoverBorderWidth: 2,
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
              display: false,
            },
            tooltip: {
              backgroundColor: "var(--color-ctp-crust)",
              titleColor: "var(--color-ctp-blue)",
              bodyColor: "var(--color-ctp-text)",
              borderColor: "var(--color-ctp-surface1)",
              borderWidth: 1,
              cornerRadius: 6,
              padding: 10,
              displayColors: false,
              titleFont: {
                size: 14,
                weight: 'bold',
              },
              bodyFont: {
                size: 13,
              },
              callbacks: {
                title: function(tooltipItems) {
                  return `Step ${tooltipItems[0].label}`;
                },
                label: function(context) {
                  return `${context.dataset.label}: ${context.formattedValue}`;
                }
              }
            },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: "Step",
                font: {
                  size: 13,
                  weight: 500,
                  family: "system-ui, sans-serif",
                },
                color: "var(--color-ctp-lavender)",
                padding: 10,
              },
              grid: {
                display: true,
                color: "rgba(180, 190, 254, 0.1)",
                tickBorderDash: [2, 4],
              },
              border: {
                display: false,
              },
              ticks: {
                color: "var(--color-ctp-subtext0)",
                font: {
                  size: 12,
                  family: "system-ui, sans-serif",
                },
                padding: 8,
                maxRotation: 0,
              },
            },
            y: {
              title: {
                display: true,
                text: label,
                font: {
                  size: 13,
                  weight: 500,
                  family: "system-ui, sans-serif",
                },
                color: "var(--color-ctp-lavender)",
                padding: 10,
              },
              grid: {
                display: true,
                color: "rgba(180, 190, 254, 0.1)",
                tickBorderDash: [2, 4],
              },
              border: {
                display: false,
              },
              ticks: {
                color: "var(--color-ctp-subtext0)",
                font: {
                  size: 12,
                  family: "system-ui, sans-serif",
                },
                padding: 8,
                precision: 4,
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
    isLoading = true;
    
    try {
      const metrics = (await loadMetrics()) as Metric[];
      const loss = Object.groupBy(metrics, ({ name }) => name);
      const chart_targets = loss[metric];

      if (chart_targets) {
        chart_targets.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
        const steps = chart_targets.map((l) => l.step ?? 0);
        const values = chart_targets.map((l) => l.value);
        createChart(metric, steps, values);
      }
    } catch (error) {
      console.error("Error displaying chart:", error);
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="p-5 space-y-4">
  {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div class="flex flex-wrap gap-2 mb-4">
      {#each experiment.availableMetrics as metric}
        <button
          class="px-3 py-1.5 text-sm font-medium rounded-md
                 transition-all duration-150 ease-in-out focus-visible:ring-2 focus-visible:ring-ctp-blue/30
                 {selectedMetric === metric
            ? 'bg-ctp-blue text-ctp-base shadow-sm'
            : 'bg-ctp-mantle text-ctp-subtext1 border border-ctp-surface0 hover:border-ctp-blue/50 hover:text-ctp-blue'}"
          onclick={() => setSelectedMetric(metric)}
        >
          {metric}
        </button>
      {/each}
    </div>
  {/if}

  {#if selectedMetric}
    <div
      class="relative h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle overflow-hidden shadow-sm"
    >
      {#if isLoading}
        <div class="absolute inset-0 flex items-center justify-center bg-ctp-mantle/80 backdrop-blur-sm z-10">
          <div class="animate-pulse text-ctp-lavender">Loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0 p-2">
        <canvas bind:this={chartCanvas}></canvas>
      </div>
    </div>
  {:else if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div class="flex flex-col items-center justify-center h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle p-8">
      <BarChart4 size={32} class="text-ctp-overlay0 mb-4" />
      <p class="text-ctp-subtext0 text-sm text-center max-w-md">
        Select a metric from above to view the chart data
      </p>
    </div>
  {/if}
</div>

<style>
  canvas {
    background-image: 
      linear-gradient(rgba(180, 190, 254, 0.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(180, 190, 254, 0.05) 1px, transparent 1px);
    background-size: 20px 20px;
    background-position: -1px -1px;
    background-color: transparent;
    border-radius: 4px;
  }
</style>