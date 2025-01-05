<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Minimize2, X } from "lucide-svelte";

  let {
    experiment,
    toggleToggleId,
  }: { experiment: Experiment; toggleToggleId: (id: number) => void } =
    $props();
</script>

<div class="flex flex-row justify-between items-center">
  <h3 class="font-medium text-lg text-gray-900 text-left">
    {experiment.name}
  </h3>
  <div class="flex flex-row gap-2">
    <button
      onclick={() => {
        toggleToggleId(experiment.id);
      }}
    >
      <Minimize2 class="w-5 h-5 text-gray-400 hover:text-gray-600" />
    </button>
    <form class="w-5 h-5" method="POST" action="?/delete">
      <input type="hidden" name="id" value={experiment.id} />
      <button type="submit">
        <X class="text-red-400 hover:text-red-600" />
      </button>
    </form>
  </div>
</div>
<div class="text-gray-400 leading-relaxed">
  <p>
    {experiment.description}
  </p>
</div>
<div class="flex flex-col gap-2">
  {#if experiment?.hyperparams}
    {#each experiment.hyperparams as hyperparam}
      <div>
        <span class="text-gray-600">{hyperparam.key}: </span>
        <span class="text-gray-400">{hyperparam.value}</span>
      </div>
    {/each}
  {/if}
</div>
{#if experiment?.groups}
  <div class="flex flex-row gap-1 text-gray-500">
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
<!-- <InteractiveChart /> -->
