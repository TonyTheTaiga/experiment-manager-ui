<script lang="ts">
  import type { HyperParam } from "$lib/types";
  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();
  let pairs = $state<HyperParam[]>([]);
</script>

<form method="POST" action="?/create" class="flex flex-col gap-6">
  <!-- Name Input -->
  <div class="space-y-2">
    <label class="text-sm font-medium text-gray-700" for="name"> Name </label>
    <input
      name="experiment-name"
      type="text"
      class="w-full px-4 py-2 bg-gray-50 border-0
             text-gray-900 placeholder-gray-400
             focus:outline-none focus:ring-0 focus:bg-gray-100
             transition-colors"
    />
  </div>

  <!-- Description Input -->
  <div class="space-y-2">
    <label class="text-sm font-medium text-gray-700" for="description">
      Description
    </label>
    <input
      name="experiment-description"
      type="text"
      class="w-full px-4 py-2 bg-gray-50 border-0
             text-gray-900 placeholder-gray-400
             focus:outline-none focus:ring-0 focus:bg-gray-100
             transition-colors"
    />
  </div>

  <!-- Hyperparameters Section -->
  <div class="space-y-3">
    {#each pairs as pair, i}
      <div class="flex gap-2 items-center">
        <input
          class="flex-1 px-4 py-2 bg-gray-50 border-0
                 text-gray-900 placeholder-gray-400
                 focus:outline-none focus:ring-0 focus:bg-gray-100
                 transition-colors"
          name="hyperparams.{i}.key"
          bind:value={pair.key}
          placeholder="Parameter name"
          required
        />
        <input
          class="flex-1 px-4 py-2 bg-gray-50 border-0
                 text-gray-900 placeholder-gray-400
                 focus:outline-none focus:ring-0 focus:bg-gray-100
                 transition-colors"
          name="hyperparams.{i}.value"
          bind:value={pair.value}
          placeholder="Value"
          required
        />
        {#if pairs.length > 0}
          <button
            type="button"
            class="text-gray-400 hover:text-gray-600 text-lg px-2"
            onclick={() => pairs.splice(i, 1)}
          >
            ×
          </button>
        {/if}
      </div>
    {/each}

    <button
      type="button"
      class="text-sm text-gray-500 hover:text-gray-700 transition-colors"
      onclick={() => (pairs = [...pairs, { key: "", value: "" }])}
    >
      + Add Parameter
    </button>
  </div>

  <!-- Action Buttons -->
  <div class="flex justify-end gap-3 pt-6 mt-6 border-t border-gray-100">
    <button
      onclick={toggleIsOpen}
      type="button"
      class="px-5 py-2 text-sm text-gray-600 bg-gray-50
             hover:bg-gray-100 transition-colors"
    >
      Cancel
    </button>
    <button
      type="submit"
      class="px-5 py-2 text-sm text-gray-700 bg-gray-200
             hover:bg-gray-300 transition-colors"
    >
      Create
    </button>
  </div>
</form>
