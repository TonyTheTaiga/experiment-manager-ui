<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Maximize2 } from "lucide-svelte";

  let {
    experiment,
    toggleToggleId,
  }: { experiment: Experiment; toggleToggleId: (id: string) => void } =
    $props();
</script>

<div class="flex flex-col gap-4">
  <!-- Header -->
  <div class="flex justify-between items-center">
    <h3 class="font-medium text-lg text-gray-900">
      {experiment.name}
    </h3>
    <button
      onclick={() => toggleToggleId(experiment.id)}
      class="text-gray-400 hover:text-gray-600 transition-colors"
    >
      <Maximize2 size={16} />
    </button>
  </div>

  <!-- Description -->
  <p class="text-gray-500 text-sm leading-relaxed">
    {experiment.description}
  </p>

  <!-- Tags -->
  {#if experiment.tags && experiment.tags.length > 0}
    <div class="flex flex-wrap items-center gap-2">
      <span class="text-sm text-gray-600">Tags:</span>
      {#each experiment.tags as tag}
        <span class="px-2 py-1 text-xs bg-gray-50 text-gray-600 rounded-sm">
          {tag}
        </span>
      {/each}
    </div>
  {/if}

  <!-- Created At -->
  {#if experiment?.createdAt}
    <time class="text-xs text-gray-400">
      Created {experiment.createdAt.toLocaleString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "numeric",
      })}
    </time>
  {/if}
</div>
