<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";
  import { Plus } from "lucide-svelte";
  import Logo from "$lib/components/logo.svelte";

  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = $state(data.experiments);

  let isOpen: boolean = $state(false);
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
      class="px-6 py-4 flex flex-row justify-between bg-ctp-mantle border-b border-ctp-surface0"
    >
      <div class="w-42 text-ctp-mauve fill-current">
        <Logo />
      </div>
      <button
        onclick={() => {
          isOpen = true;
        }}
        class="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-ctp-mauve text-ctp-crust hover:bg-ctp-lavender transition-colors font-medium"
      >
        <Plus size={16} /> New Experiment</button
      >
    </nav>
  </header>

  <main class="p-4 flex-1">
    <ExperimentsList bind:experiments />
  </main>
</div>
