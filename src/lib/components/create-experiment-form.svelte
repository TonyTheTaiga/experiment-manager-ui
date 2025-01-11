<script lang="ts">
  import type { HyperParam } from "$lib/types";
  import { SquarePlus, CirclePlus } from "lucide-svelte";

  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();

  let pairs = $state<HyperParam[]>([]);

  // Tags stuff
  let addingNewTag = $state<boolean>(false);
  let tag = $state<string | null>(null);
  let tags = $state<string[]>([]);

  function addTag() {
    if (tag) {
      tags = [...tags, tag];
      tag = null;
      addingNewTag = false;
    }
  }
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

  <!-- Tags -->
  <div class="flex flex-wrap items-center gap-3">
    <span class="text-sm font-medium text-gray-700">Tags</span>

    {#each tags as tag, i}
      <span
        class="px-3 py-1 text-sm bg-gray-50 text-gray-600
                flex items-center gap-1 group"
      >
        {tag}
        <button
          class="text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity"
          onclick={() => tags.splice(i, 1)}
        >
          ×
        </button>
      </span>
    {/each}

    {#if addingNewTag}
      <div class="flex items-center gap-2">
        <input
          type="text"
          bind:value={tag}
          class="px-3 py-1 w-32 text-sm bg-gray-50
               text-gray-900 placeholder-gray-400
               focus:outline-none focus:ring-0 focus:bg-gray-100
               transition-colors"
          placeholder="New tag"
          onkeydown={(e) => {
            if (e.key === "Enter") {
              addTag();
            }
          }}
        />
        <button
          onclick={(e) => {
            e.preventDefault();
            addTag();
          }}
          class="text-gray-400 hover:text-gray-600 transition-colors"
        >
          <CirclePlus size={16} />
        </button>
      </div>
    {/if}

    {#if !addingNewTag}
      <button
        onclick={() => {
          addingNewTag = true;
        }}
        class="text-gray-400 hover:text-gray-600 transition-colors"
      >
        <SquarePlus size={16} />
      </button>
    {/if}
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
          placeholder="Parameter name"
          required
        />
        <input
          class="flex-1 px-4 py-2 bg-gray-50 border-0
                 text-gray-900 placeholder-gray-400
                 focus:outline-none focus:ring-0 focus:bg-gray-100
                 transition-colors"
          name="hyperparams.{i}.value"
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
