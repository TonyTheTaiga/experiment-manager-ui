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

<header>
  <nav
    class="px-6 py-4 flex flex-row justify-end bg-white border-b border-gray-200"
  >
    <button
      onclick={() => {
        isOpen = true;
      }}
      class="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-s-full transition-colors"
      >(+) Experiment</button
    >
  </nav>
</header>

<main class="mx-4 my-4">
  <ExperimentsList {experiments} />
</main>

<style lang="postcss">
  :global(html) {
    background-color: theme(colors.gray.100);
  }
</style>
