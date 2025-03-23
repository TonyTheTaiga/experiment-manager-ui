<script lang="ts">
  import type { HyperParam } from "$lib/types";
  import { Plus, X, Tag as TagIcon, Settings } from "lucide-svelte";

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
    <label class="text-sm font-medium text-ctp-text" for="name">Experiment Name</label>
    <input
      name="experiment-name"
      type="text"
      class="w-full px-4 py-2.5 bg-ctp-mantle border border-ctp-surface0 rounded-md
             text-ctp-text placeholder-ctp-subtext0
             focus:outline-none focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue/20
             transition-colors"
      placeholder="Enter experiment name"
    />
  </div>

  <!-- Description Input -->
  <div class="space-y-2">
    <label class="text-sm font-medium text-ctp-text" for="description">
      Description
    </label>
    <textarea
      name="experiment-description"
      rows="3"
      class="w-full px-4 py-2.5 bg-ctp-mantle border border-ctp-surface0 rounded-md
             text-ctp-text placeholder-ctp-subtext0
             focus:outline-none focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue/20
             transition-colors resize-none"
      placeholder="Briefly describe this experiment"
    ></textarea>
  </div>

  <!-- Tags Section -->
  <div class="space-y-3">
    <div class="flex items-center gap-2 text-ctp-lavender">
      <TagIcon size={16} />
      <h3 class="text-sm font-medium">Tags</h3>
    </div>
    
    <div class="flex flex-wrap items-center gap-2">
      {#each tags as tag, i}
        <input type="hidden" value={tag} name="tags.{i}" />
        <span
          class="px-2.5 py-1 text-xs bg-ctp-mantle text-ctp-lavender rounded-full border border-ctp-surface0
              flex items-center gap-1.5 group"
        >
          {tag}
          <button
            type="button"
            class="text-ctp-subtext0 hover:text-ctp-red transition-colors"
            onclick={() => tags.splice(i, 1)}
          >
            <X size={12} />
          </button>
        </span>
      {/each}

      {#if addingNewTag}
        <div class="flex items-center gap-1">
          <input
            type="text"
            bind:value={tag}
            class="px-3 py-1 w-32 text-xs bg-ctp-mantle border border-ctp-surface0 rounded-md
                 text-ctp-text placeholder-ctp-subtext0
                 focus:outline-none focus:border-ctp-lavender
                 transition-colors"
            placeholder="New tag"
            onkeydown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                addTag();
              }
            }}
          />
          <button
            type="button"
            onclick={(e) => {
              e.preventDefault();
              addTag();
            }}
            class="p-1 text-ctp-subtext0 hover:text-ctp-lavender hover:bg-ctp-surface0 rounded transition-colors"
          >
            <Plus size={14} />
          </button>
        </div>
      {:else}
        <button
          type="button"
          onclick={(e) => {
            e.preventDefault();
            addingNewTag = true;
          }}
          class="px-2.5 py-1 text-xs bg-ctp-mantle text-ctp-subtext0 rounded-full border border-ctp-surface0
                hover:text-ctp-lavender hover:border-ctp-lavender/30 transition-colors flex items-center gap-1"
        >
          <Plus size={12} />
          Add Tag
        </button>
      {/if}
    </div>
  </div>

  <!-- Hyperparameters Section -->
  <div class="space-y-3">
    <div class="flex items-center gap-2 text-ctp-blue">
      <Settings size={16} />
      <h3 class="text-sm font-medium">Parameters</h3>
    </div>

    {#if pairs.length === 0}
      <div class="text-xs text-ctp-subtext0 bg-ctp-mantle p-3 rounded-md border border-ctp-surface0">
        No parameters defined yet. Add parameters to track experiment configuration values.
      </div>
    {/if}

    {#each pairs as pair, i}
      <div class="flex gap-2 items-center">
        <input
          class="flex-1 px-3 py-2 bg-ctp-mantle border border-ctp-surface0 rounded-md
                 text-sm text-ctp-text placeholder-ctp-subtext0
                 focus:outline-none focus:border-ctp-blue
                 transition-colors"
          name="hyperparams.{i}.key"
          placeholder="Parameter name"
          required
        />
        <input
          class="flex-1 px-3 py-2 bg-ctp-mantle border border-ctp-surface0 rounded-md
                 text-sm text-ctp-text placeholder-ctp-subtext0
                 focus:outline-none focus:border-ctp-blue
                 transition-colors"
          name="hyperparams.{i}.value"
          placeholder="Value"
          required
        />
        <button
          type="button"
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0 rounded transition-colors"
          onclick={() => pairs.splice(i, 1)}
        >
          <X size={16} />
        </button>
      </div>
    {/each}

    <button
      type="button"
      class="px-3 py-1.5 text-xs bg-ctp-mantle text-ctp-subtext0 rounded border border-ctp-surface0
            hover:text-ctp-blue hover:border-ctp-blue/30 transition-colors flex items-center gap-1.5"
      onclick={() => (pairs = [...pairs, { key: "", value: "" }])}
    >
      <Plus size={12} />
      Add Parameter
    </button>
  </div>

  <!-- Action Buttons -->
  <div class="flex justify-end gap-3 pt-4 mt-2 border-t border-ctp-surface0">
    <button
      onclick={toggleIsOpen}
      type="button"
      class="px-5 py-2 text-sm font-medium text-ctp-subtext1 bg-ctp-mantle border border-ctp-surface0 rounded-md
             hover:text-ctp-text hover:bg-ctp-crust transition-colors"
    >
      Cancel
    </button>
    <button
      type="submit"
      class="px-5 py-2 text-sm font-medium text-ctp-mantle bg-ctp-blue rounded-md shadow-sm
             hover:bg-ctp-blue/90 transition-colors flex items-center gap-2"
    >
      <Plus size={16} />
      Create Experiment
    </button>
  </div>
</form>