<script lang="ts">
  import CreateExperimentForm from "./create-experiment-form.svelte";
  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();
</script>

<div
  class="fixed inset-0 bg-black/50 backdrop-blur-[2px]
         flex items-center justify-center p-4"
>
  <div class="bg-gray-800 w-full max-w-xl border border-gray-700">
    <div class="px-8 pt-8 pb-6">
      <h2 class="text-xl font-medium text-gray-100">Create a New Experiment</h2>
    </div>

    <div class="px-8 pb-8">
      <CreateExperimentForm {toggleIsOpen} />
    </div>
  </div>
</div>
