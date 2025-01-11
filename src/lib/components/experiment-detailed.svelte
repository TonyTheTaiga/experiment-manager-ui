<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Minimize2, X } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";

  export let experiment: Experiment;
  export let toggleToggleId: (id: string) => void;
</script>

<article class="p-4 bg-white">
  <header class="flex justify-between items-center">
    <time class="text-sm text-gray-400">
      {new Date(experiment.createdAt).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "numeric",
      })}
    </time>
    <div class="flex items-center gap-3">
      <button
        onclick={() => toggleToggleId(experiment.id)}
        class="text-gray-600 hover:text-black transition-colors flex items-center justify-center"
      >
        <Minimize2 size={16} />
      </button>
      <form method="POST" action="?/delete" class="flex items-center">
        <input type="hidden" name="id" value={experiment.id} />
        <button
          type="submit"
          class="text-gray-600 hover:text-red-600 transition-colors flex items-center justify-center"
        >
          <X size={16} />
        </button>
      </form>
    </div>
  </header>

  <h2 class="text-2xl font-medium text-gray-900 mb-6">
    {experiment.name}
  </h2>

  <p class="text-sm text-gray-500 mb-2 leading-relaxed">
    {experiment.description}
  </p>

  {#if experiment.tags && experiment.tags.length > 0}
    <div class="flex flex-wrap items-center gap-2 mb-2">
      <span class="text-sm text-gray-600">Tags:</span>
      {#each experiment.tags as tag}
        <span class="px-2 py-1 text-xs bg-gray-50 text-gray-600 rounded-sm">
          {tag}
        </span>
      {/each}
    </div>
  {/if}

  {#if experiment.hyperparams}
    <div class="flex flex-wrap gap-4 mb-6">
      {#each experiment.hyperparams as param}
        <div class="flex items-center gap-1">
          <span class="text-xs text-gray-600">{param.key}</span>
          <span class="text-xs text-gray-400">{param.value}</span>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Metrics -->
  {#if experiment.availableMetrics}
    <div class="mt-4">
      <InteractiveChart {experiment} />
    </div>
  {/if}
</article>
