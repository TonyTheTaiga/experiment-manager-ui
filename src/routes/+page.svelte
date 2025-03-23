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
    class="px-6 py-4 flex flex-row justify-end bg-ctp-mantle border-b border-ctp"
  >
    <button
      onclick={() => {
        isOpen = true;
      }}
      class="px-4 py-2 bg-ctp-mauve hover:bg-ctp-lavender text-ctp-mantle rounded-s-full transition-colors"
      >(+) Experiment</button
    >
  </nav>
</header>

<main class="mx-4 my-4">
  <ExperimentsList {experiments} />
</main>