<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Maximize2 } from "lucide-svelte";

  let {
    experiment,
    toggleToggleId,
  }: { experiment: Experiment; toggleToggleId: (id: string) => void } =
    $props();
</script>

<div class="flex flex-row justify-between">
  <h3 class="font-medium text-lg text-gray-900">
    {experiment.name}
  </h3>
  <button
    onclick={() => {
      toggleToggleId(experiment.id);
    }}
  >
    <Maximize2 class="text-gray-400 hover:text-gray-600" size={16} />
  </button>
</div>

<p class="text-gray-400 text-sm leading-relaxed">
  {experiment.description}
</p>
{#if experiment?.createdAt}
  <time class="text-gray-800 text-xs mt-auto pt-4">
    Created: {experiment.createdAt.toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
    })}
  </time>
{/if}
