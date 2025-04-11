<script lang="ts">
  import type { Experiment } from "$lib/types";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import { Cpu } from "lucide-svelte";

  const { experiments = $bindable() }: { experiments: Experiment[] } = $props();
  let selectedId = $state<string | null>(null);
  let highlighted = $state<string[]>([]);
</script>

<section>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    {#each experiments as experiment, idx (experiment.id)}
      <div
        class="
            rounded-lg border border-[var(--color-ctp-surface1)] overflow-hidden
            {selectedId === experiment.id
          ? 'md:col-span-2 lg:col-span-4 row-span-2 order-first'
          : 'order-none bg-[var(--color-ctp-base)] hover:shadow-md transition-shadow duration-200'}
            {highlighted.length > 0 && !highlighted.includes(experiment.id)
          ? 'opacity-40'
          : ''}
          "
      >
        {#if selectedId !== experiment.id}
          <ExperimentSimple bind:selectedId bind:highlighted {experiment} />
        {:else}
          <ExperimentDetailed
            bind:selectedId
            bind:highlighted
            bind:experiment={experiments[idx]}
          />
        {/if}
      </div>
    {/each}
  </div>

  {#if experiments.length === 0}
    <div
      class="flex flex-col items-center justify-center p-12 text-center bg-[var(--color-ctp-mantle)] rounded-lg border border-[var(--color-ctp-surface1)]"
    >
      <Cpu
        size={40}
        className="text-[var(--color-ctp-overlay0)] mb-4"
        strokeWidth={1.5}
      />
      <h3 class="text-lg font-medium text-[var(--color-ctp-text)] mb-2">
        No experiments yet
      </h3>
      <p class="text-[var(--color-ctp-subtext0)] max-w-md">
        Create your first experiment to start tracking metrics and see them
        displayed here.
      </p>
    </div>
  {/if}
</section>
