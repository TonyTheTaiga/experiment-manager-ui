<script lang="ts">
  import type { Experiment } from "$lib/types";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import ExperimentSimple from "./experiment-simple.svelte";

  let { experiments }: { experiments: Experiment[] } = $props();

  let selectedId = $state<string | null>(null);
  function toggleToggleId(id: string) {
    if (selectedId === id) {
      selectedId = null;
    } else {
      selectedId = id;
    }
  }
</script>

<section>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    {#each experiments as experiment}
      <div
        class={`
                    bg-gray-800 rounded-sm p-4
                    ${
                      selectedId === experiment.id
                        ? "md:col-span-2 lg:col-span-4 row-span-2 order-first"
                        : "order-none"
                    }
                `}
      >
        <article class="flex flex-col gap-1">
          {#if selectedId !== experiment.id}
            <ExperimentSimple {experiment} {toggleToggleId} />
          {:else}
            <ExperimentDetailed {experiment} {toggleToggleId} />
          {/if}
        </article>
      </div>
    {/each}
  </div>
</section>
