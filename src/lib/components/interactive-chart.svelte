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
              borderColor: "#74c7ec", /* sapphire */
              backgroundColor: "rgba(116, 199, 236, 0.15)",
              fill: true,
              pointBackgroundColor: "#b4befe", /* lavender */
              pointBorderColor: "#181825", /* mantle */
              pointHoverBackgroundColor: "#cba6f7", /* mauve */
              pointHoverBorderColor: "#1e1e2e", /* base */
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'nearest',
            intersect: false,
            axis: 'x',
            includeInvisible: true,
          },
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              backgroundColor: "#11111b", /* crust */
              titleColor: "#74c7ec", /* sapphire */
              bodyColor: "#cdd6f4", /* text */
              borderColor: "#6c7086", /* overlay0 */
              position: 'nearest',
              caretPadding: 10,
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
                color: "#cdd6f4", /* text */
              },
              ticks: {
                color: "#cdd6f4", /* text */
              },
              grid: {
                color: "rgba(180, 190, 254, 0.08)",
              },
            },
            y: {
              title: {
                display: true,
                text: label,
                color: "#cdd6f4", /* text */
              },
              ticks: {
                color: "#cdd6f4", /* text */
              },
              grid: {
                color: "rgba(180, 190, 254, 0.08)",
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

      if (chart_targets && chart_targets.length > 0) {
        // Sort metrics by step
        chart_targets.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
        
        // Handle missing steps by using indices if step is undefined
        const steps = chart_targets.map((l, index) => l.step !== undefined ? l.step : index);
        
        // Ensure all values are numbers
        const values = chart_targets.map((l) => typeof l.value === 'number' ? l.value : parseFloat(l.value) || 0);
        
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
          class={`py-1.5 px-3 text-sm font-medium rounded-md transition-colors ${
            selectedMetric === metric 
              ? 'bg-[var(--color-ctp-mauve)] text-[var(--color-ctp-crust)] hover:bg-[var(--color-ctp-lavender)]' 
              : 'bg-[var(--color-ctp-surface0)] text-[var(--color-ctp-text)] border border-[var(--color-ctp-surface1)] hover:bg-[var(--color-ctp-blue)] hover:text-[var(--color-ctp-crust)] hover:border-[var(--color-ctp-blue)]'
          }`}
          onclick={() => setSelectedMetric(metric)}
        >
          {metric}
        </button>
      {/each}
    </div>
  {/if}

  {#if selectedMetric}
    <div
      class="relative h-80 w-full rounded-md border border-[var(--color-ctp-surface1)] bg-[var(--color-ctp-mantle)] overflow-hidden shadow-md"
    >
      {#if isLoading}
        <div class="absolute inset-0 flex items-center justify-center bg-[var(--color-ctp-mantle)]/80 backdrop-blur-sm z-10">
          <div class="animate-pulse text-[#89dceb]">Loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0 p-4">
        <canvas bind:this={chartCanvas}></canvas>
      </div>
    </div>
  {:else if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div class="flex flex-col items-center justify-center h-80 w-full rounded-md border border-[var(--color-ctp-surface1)] bg-[var(--color-ctp-mantle)] p-8">
      <BarChart4 size={32} class="text-[var(--color-ctp-overlay0)] mb-4" />
      <p class="text-[var(--color-ctp-subtext0)] text-sm text-center max-w-md">
        Select a metric from above to view the chart data
      </p>
    </div>
  {/if}
</div>

<style>
  canvas {
    background-color: transparent;
    border-radius: 4px;
  }
</style>