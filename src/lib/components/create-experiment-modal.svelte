<script lang="ts">
  import CreateExperimentForm from "./create-experiment-form.svelte";
  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();
</script>

<div
  class="fixed inset-0 bg-gray-950/20 backdrop-blur-[2px]
         flex items-center justify-center p-4"
>
  <div class="bg-white w-full max-w-xl">
    <div class="px-8 pt-8 pb-6">
      <h2 class="text-xl font-medium text-gray-900">Create a New Experiment</h2>
    </div>

    <div class="px-8 pb-8">
      <CreateExperimentForm {toggleIsOpen} />
    </div>
  </div>
</div>
