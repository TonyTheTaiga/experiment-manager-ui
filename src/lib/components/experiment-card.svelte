<script lang="ts">
  import type { Experiment } from "$lib/types";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  
  let { experiment, toggleToggleId, targetId }: { 
    experiment: Experiment;
    toggleToggleId: (id: string) => void;
    targetId: string | null;
  } = $props();
</script>

<div
  class="
    rounded-lg border border-ctp-surface1 overflow-hidden
    {targetId === experiment.id
      ? 'md:col-span-2 lg:col-span-4 row-span-2 order-first'
      : 'order-none bg-ctp-base hover:shadow-md transition-shadow duration-200'
    }
  "
>
  {#if targetId !== experiment.id}
    <ExperimentSimple {experiment} {toggleToggleId} />
  {:else}
    <ExperimentDetailed {experiment} {toggleToggleId} />
  {/if}
</div>