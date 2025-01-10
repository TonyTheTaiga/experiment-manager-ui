<script lang="ts">
  import type { Experiment } from "$lib/types";
  import InteractiveChart from "./interactive-chart.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import ExperimentSimple from "./experiment-simple.svelte";

  let { experiments }: { experiments: Experiment[] } = $props();
  let targetId = $state<string | null>(null);

  function toggleToggleId(id: string) {
    targetId = targetId === id ? null : id;
  }
</script>

<section>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {#each experiments as experiment}
      <div
        class={`
                    bg-white rounded-sm p-4 
                    ${
                      targetId === experiment.id
                        ? "md:col-span-2 lg:col-span-4 row-span-2 order-first"
                        : "order-none"
                    }
                `}
      >
        <article class={"flex flex-col gap-1"}>
          {#if targetId !== experiment.id}
            <ExperimentSimple {experiment} {toggleToggleId} />
          {:else}
            <ExperimentDetailed {experiment} {toggleToggleId} />
          {/if}
        </article>
      </div>
    {/each}
  </div>
</section>
