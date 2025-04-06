<script lang="ts">
  import type { HyperParam, Experiment } from "../types";
  import { Plus, X, Tag as TagIcon, Settings } from "lucide-svelte";

  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();

  let pairs = $state<HyperParam[]>([]);
  let addingNewTag = $state<boolean>(false);
  let tag = $state<string | null>(null);
  let tags = $state<string[]>([]);
  let experimentList = $state<Experiment[]>([]);
  $inspect(experimentList);
  let reference = $state<Experiment | null>(null);

  function addTag() {
    if (tag) {
      tags = [...tags, tag];
      tag = null;
      addingNewTag = false;
    }
  }

  async function getExperiments(query: string | null) {
    console.log(query);
    let url = `/api/experiments`;
    if (query) {
      url += `?startwith=${encodeURIComponent(query)}`;
    }

    await fetch(url, { method: "GET" })
      .then((res) => res.json())
      .then((data) => {
        experimentList = data as Experiment[];
      });
  }

  const charList: string[] = [];

  async function handleKeyDown(event: KeyboardEvent) {
    const input = event.target as HTMLInputElement;

    if (event.key === "Backspace") {
      const { selectionStart, selectionEnd } = input;

      if (selectionStart !== null && selectionEnd !== null) {
        const deleteCount =
          selectionStart !== selectionEnd ? selectionEnd - selectionStart : 1;
        const deleteIndex =
          selectionStart !== selectionEnd ? selectionStart : selectionStart - 1;

        if (deleteIndex >= 0) charList.splice(deleteIndex, deleteCount);
      } else {
        charList.pop();
      }
    } else if (/^[a-z0-9]$/i.test(event.key)) {
      charList.push(event.key);
    }

    if (charList.length && charList.length % 2 === 0) {
      await getExperiments(charList.join(""));
      console.log("charList", charList);
    } else if (charList.length === 0) {
      experimentList = [];
    }
  }
</script>

<form method="POST" action="?/create" class="flex flex-col gap-6">
  <!-- Name Input -->
  <div class="space-y-3">
    <label class="text-sm font-medium text-[var(--color-ctp-text)]" for="name"
      >Experiment Name</label
    >
    <input
      name="experiment-name"
      type="text"
      class="w-full px-3.5 py-2.5 bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors"
      placeholder="Enter experiment name"
    />
  </div>

  <!-- Description Input -->
  <div class="space-y-3">
    <label
      class="text-sm font-medium text-[var(--color-ctp-text)]"
      for="description"
    >
      Description
    </label>
    <textarea
      name="experiment-description"
      rows="3"
      class="w-full px-3.5 py-2.5 bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors resize-none"
      placeholder="Briefly describe this experiment"
    ></textarea>
  </div>

  <!-- Tags Section -->
  <div class="space-y-3">
    <div class="flex items-center gap-2">
      <TagIcon size={16} class="text-[var(--color-ctp-mauve)]" />
      <h3 class="text-lg font-semibold text-[var(--color-ctp-blue)]">Tags</h3>
    </div>

    <div class="flex flex-wrap items-center gap-2">
      {#each tags as tag, i}
        <input type="hidden" value={tag} name="tags.{i}" />
        <span
          class="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-[var(--color-ctp-surface0)] text-[var(--color-ctp-mauve)] border border-[var(--color-ctp-surface0)] group"
        >
          {tag}
          <button
            type="button"
            class="text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-red)] transition-colors ml-1.5"
            onclick={() => tags.splice(i, 1)}
            aria-label="Remove tag"
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
            class="w-32 px-3 py-1 text-xs bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors"
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
            class="p-1.5 rounded-full text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] transition-colors"
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
          class="inline-flex items-center gap-1 py-0.5 px-2 text-xs rounded-full bg-transparent text-[var(--color-ctp-mauve)] border border-[var(--color-ctp-mauve)] hover:bg-[var(--color-ctp-mauve)]/10 transition-colors"
        >
          <Plus size={12} />
          Add Tag
        </button>
      {/if}
    </div>
  </div>

  <!-- Hyperparameters Section -->
  <div class="space-y-3">
    <div class="flex items-center gap-2">
      <Settings size={16} class="text-[var(--color-ctp-mauve)]" />
      <h3 class="text-lg font-semibold text-[var(--color-ctp-blue)]">
        Parameters
      </h3>
    </div>

    {#each pairs as pair, i}
      <div class="flex gap-2 items-center">
        <input
          class="w-full px-3.5 py-2 text-sm bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors flex-1"
          name="hyperparams.{i}.key"
          placeholder="Parameter name"
          required
        />
        <input
          class="w-full px-3.5 py-2 text-sm bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors flex-1"
          name="hyperparams.{i}.value"
          placeholder="Value"
          required
        />
        <button
          type="button"
          class="p-1.5 text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-red)] hover:bg-[var(--color-ctp-surface0)] rounded transition-colors"
          onclick={() => pairs.splice(i, 1)}
        >
          <X size={16} />
        </button>
      </div>
    {/each}

    <button
      type="button"
      class="inline-flex items-center gap-1.5 py-1.5 px-3 text-sm font-medium rounded-md bg-transparent text-[var(--color-ctp-mauve)] border border-[var(--color-ctp-mauve)] hover:bg-[var(--color-ctp-mauve)]/10 transition-colors"
      onclick={() => (pairs = [...pairs, { key: "", value: "" }])}
    >
      <Plus size={12} />
      Add Parameter
    </button>
  </div>

  <!-- References -->
  <div class="space-y-4">
    <div class="flex items-center gap-2">
      <Settings size={16} class="text-[var(--color-ctp-mauve)]" />
      <h3 class="text-lg font-semibold text-[var(--color-ctp-blue)]">
        References
      </h3>
    </div>

    <div>
      {#if reference}
        {reference.name}
      {/if}
    </div>

    <div>
      <div
        class="flex flex-col space-y-2 p-2 border border-[var(--color-ctp-surface0)] rounded-md"
      >
        <input
          id="search-input"
          placeholder="Search for references..."
          class="w-full px-2.5 py-1.5 bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors"
          onkeydown={async (event) => await handleKeyDown(event)}
        />
        {#if experimentList.length > 0}
          <ul>
            {#each experimentList as experiment}
              <button
                onclick={(event) => {
                  event.preventDefault();
                  reference = experiment;
                }}
              >
                {experiment.name}
              </button>
            {/each}
          </ul>
        {/if}
      </div>
    </div>
  </div>

  <!-- Action Buttons -->
  <div
    class="flex justify-end gap-3 pt-4 mt-2 border-t border-[var(--color-ctp-surface0)]"
  >
    <button
      onclick={toggleIsOpen}
      type="button"
      class="inline-flex items-center justify-center px-4 py-2 font-medium rounded-md bg-[var(--color-ctp-surface0)] text-[var(--color-ctp-text)] border border-[var(--color-ctp-surface1)] hover:bg-[var(--color-ctp-surface1)] transition-colors"
    >
      Cancel
    </button>
    <button
      type="submit"
      class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-md bg-[var(--color-ctp-mauve)] text-[var(--color-ctp-crust)] hover:bg-[var(--color-ctp-lavender)] transition-colors"
    >
      <Plus size={16} />
      Create Experiment
    </button>
  </div>
</form>
