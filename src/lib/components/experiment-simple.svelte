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
    <Maximize2 class="w-5 h-5 text-gray-400 hover:text-gray-600" />
  </button>
</div>

<p class="text-gray-400 text-sm leading-relaxed">
  {experiment.description}
</p>
{#if experiment?.groups}
  <div class="flex flex-row gap-1 text-sm text-gray-500">
    <span>Groups:</span>
    <ul class="flex flex-row gap-1">
      {#each experiment.groups as group}
        <li class="items-center">
          <span>{group}</span>
        </li>
      {/each}
    </ul>
  </div>
{/if}
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
