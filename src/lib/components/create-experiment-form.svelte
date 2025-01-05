<script lang="ts">
  import type { HyperParam } from "$lib/types";

  let { toggleIsOpen }: { toggleIsOpen: () => void } = $props();
  let pairs = $state<HyperParam[]>([]);
</script>

<form method="POST" action="?/create" class="p-4">
  <div class="flex flex-col gap-2">
    <div>
      <label class="block text-sm text-gray-700" for="name">Name</label>
      <input
        name="experiment-name"
        type="text"
        class="w-full px-3 py-2 border border-gray-300 rounded-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
    </div>

    <div>
      <label class="block text-sm text-gray-700" for="name">Description</label>
      <input
        name="experiment-description"
        type="text"
        class="w-full px-3 py-2 border border-gray-300 rounded-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
    </div>

    <div class="flex flex-col">
      <!-- <span class="text-gray-500"> Hyperparams: </span> -->
      {#each pairs as pair, i}
        <div class="pair">
          <input
            class="px-3 py-2"
            name="hyperparams.{i}.key"
            bind:value={pair.key}
            placeholder="Name"
            required
          />
          <input
            class="px-3 py-2"
            name="hyperparams.{i}.value"
            bind:value={pair.value}
            placeholder="Value"
            required
          />
          {#if pairs.length > 0}
            <button type="button" onclick={() => pairs.splice(i, 1)}>×</button>
          {/if}
        </div>
      {/each}
      <div class="justify-start">
        <button
          class="px-1 py-1 rounded-sm text-sm text-blue-300"
          type="button"
          onclick={() => (pairs = [...pairs, { key: "", value: "" }])}
          >Add Hyper Parameter</button
        >
      </div>
    </div>

    <div class="pt-2 border-t border-gray-200 flex gap-2 justify-end">
      <button
        type="submit"
        class="px-4 py-2 bg-gray-300 text-gray-700 rounded-sm hover:bg-gray-400 transition-colors"
      >
        Submit
      </button>
      <button
        onclick={toggleIsOpen}
        class="px-4 py-2 bg-gray-100 text-gray-700 rounded-sm hover:bg-gray-200 transition-colors"
      >
        Close
      </button>
    </div>
  </div>
</form>
