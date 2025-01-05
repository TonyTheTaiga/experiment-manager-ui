<script lang="ts">
  import CreateExperimentForm from "./create-experiment-form.svelte";

  let { toggleIsOpen }: { toggleIsOpen: () => void } = $props();
</script>

<div
  class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center"
>
  <div class="bg-white rounded-sm flex flex-col">
    <div class="p-2">
      <h2 class="text-lg font-medium">Create a New Experiment</h2>
      <div class="mt-2 border-b border-gray-200"></div>
    </div>
    <CreateExperimentForm {toggleIsOpen} />
  </div>
</div>
