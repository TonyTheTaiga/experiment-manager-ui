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
  class="fixed inset-0 bg-ctp-crust/60 backdrop-blur-sm
         flex items-center justify-center p-4 z-50"
>
  <div class="bg-ctp-base w-full max-w-xl rounded-lg border border-ctp-surface1 shadow-lg overflow-hidden">
    <div class="px-6 py-4 border-b border-ctp-surface0 flex justify-between items-center">
      <h2 class="text-xl font-medium text-ctp-text">Create Experiment</h2>
      <button 
        onclick={toggleIsOpen} 
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface0 rounded-full transition-colors"
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