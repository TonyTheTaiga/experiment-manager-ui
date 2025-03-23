<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";

  let isOpen = $state(false);
  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = data.experiments;

  function toggleIsOpen() {
    isOpen = !isOpen;
  }
</script>

{#if isOpen}
  <CreateExperimentModal {toggleIsOpen} />
{/if}

<div class="flex flex-col h-full">
  <header>
    <nav
      class="px-6 py-4 flex flex-row justify-end bg-[var(--color-ctp-mantle)] border-b border-[var(--color-ctp-surface0)]"
    >
      <button
        onclick={() => {
          isOpen = true;
        }}
        class="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-[var(--color-ctp-mauve)] text-[var(--color-ctp-crust)] hover:bg-[var(--color-ctp-lavender)] transition-colors font-medium"
        ><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg> New Experiment</button
      >
    </nav>
  </header>

  <main class="p-4 flex-1">
    <ExperimentsList {experiments} />
  </main>
</div>