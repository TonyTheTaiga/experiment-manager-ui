<script lang="ts">
  import CreateExperimentForm from "./create-experiment-form.svelte";
  import { X } from "lucide-svelte";
  
  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();
</script>

<div
  class="fixed inset-0 bg-[var(--color-ctp-crust)]/60 backdrop-blur-sm
         flex items-center justify-center p-4 z-50"
>
  <div class="bg-[var(--color-ctp-base)] w-full max-w-xl rounded-lg border border-[var(--color-ctp-surface1)] shadow-lg overflow-hidden">
    <div class="px-6 py-4 border-b border-[var(--color-ctp-surface0)] flex justify-between items-center">
      <h2 class="text-xl font-medium text-[var(--color-ctp-text)]">Create Experiment</h2>
      <button 
        onclick={toggleIsOpen} 
        class="p-1.5 text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] rounded-full transition-colors"
        aria-label="Close modal"
      >
        <X size={18} />
      </button>
    </div>

    <div class="p-6">
      <CreateExperimentForm {toggleIsOpen} />
    </div>
  </div>
</div>