<script lang="ts">
  import { X, Save, TagIcon, Plus } from "lucide-svelte";
  import { enhance } from "$app/forms";

  let { experiment = $bindable(), editMode = $bindable() } = $props();

  let addingNewTag = $state(false);
  let tag = $state<string | null>(null);

  function addTag(e: KeyboardEvent | MouseEvent) {
    e.preventDefault();
    if (tag && tag !== "") {
      experiment.tags.push(tag);
      tag = null;
    }
  }
</script>

<div
  class="fixed inset-0 bg-[var(--color-ctp-crust)]/80 backdrop-blur-md
         flex items-center justify-center p-4 z-50"
>
  <!-- MODAL CONTAINER -->
  <div
    class="bg-[var(--color-ctp-mantle)] w-full max-w-xl rounded-xl border border-[var(--color-ctp-surface0)] shadow-2xl overflow-hidden"
  >
    <!-- HEADER -->
    <div
      class="px-6 py-4 border-b border-[var(--color-ctp-surface0)] flex justify-between items-center"
    >
      <h2
        class="text-xl font-medium text-[var(--color-ctp-text)] flex items-center gap-2"
      >
        <Save size={18} class="text-[var(--color-ctp-mauve)]" />
        Edit Experiment
      </h2>
      <button
        onclick={() => {
          editMode = !editMode;
        }}
        class="p-2 text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)]/50 rounded-full transition-all"
        aria-label="Close modal"
      >
        <X size={18} />
      </button>
    </div>

    <!-- FORM -->
    <div class="p-6">
      <form
        method="POST"
        action="?/update"
        class="flex flex-col gap-8"
        use:enhance={({ formElement, formData, action, cancel, submitter }) => {
          experiment.name = formData.get("experiment-name");
          experiment.description = formData.get("experiment-description");
          return async ({ result, update }) => {
            editMode = !editMode;
          };
        }}
      >
        <input
          class="hidden"
          id="experiment-id"
          name="experiment-id"
          value={experiment.id}
        />

        <!-- Basic Info Section -->
        <div class="space-y-5">
          <!-- Name Input -->
          <div class="space-y-2">
            <label
              class="text-sm font-medium text-[var(--color-ctp-subtext0)]"
              for="name">Experiment Name</label
            >
            <input
              id="experiment-name"
              name="experiment-name"
              type="text"
              class="w-full px-4 py-3 bg-[var(--color-ctp-base)] border-0 rounded-lg text-[var(--color-ctp-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-ctp-mauve)] transition-all placeholder-[var(--color-ctp-overlay0)] shadow-sm"
              placeholder="Enter experiment name"
              value={experiment.name}
            />
          </div>

          <!-- Description Input -->
          <div class="space-y-2">
            <label
              class="text-sm font-medium text-[var(--color-ctp-subtext0)]"
              for="description"
            >
              Description
            </label>
            <textarea
              id="experiment-description"
              name="experiment-description"
              rows="3"
              class="w-full px-4 py-3 bg-[var(--color-ctp-base)] border-0 rounded-lg text-[var(--color-ctp-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-ctp-blue)] transition-all resize-none placeholder-[var(--color-ctp-overlay0)] shadow-sm"
              placeholder="Briefly describe this experiment"
              value={experiment.description}
            ></textarea>
          </div>
        </div>

        <!-- Tags Section -->
        <div class="space-y-4">
          <div
            class="flex items-center gap-3 pb-2 border-b border-[var(--color-ctp-surface0)]"
          >
            <TagIcon size={18} class="text-[var(--color-ctp-pink)]" />
            <h3 class="text-xl font-medium text-[var(--color-ctp-text)]">
              Tags
            </h3>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            {#each experiment.tags as tag, i}
              <input type="hidden" value={tag} name="tags.{i}" />
              <span
                class="inline-flex items-center px-3 py-1 text-xs font-medium rounded-full bg-[var(--color-ctp-mauve)]/10 text-[var(--color-ctp-mauve)] border-0 group"
              >
                {tag}
                <button
                  type="button"
                  class="text-[var(--color-ctp-mauve)]/70 hover:text-[var(--color-ctp-red)] transition-colors ml-2"
                  onclick={() => experiment.tags.splice(i, 1)}
                  aria-label="Remove tag"
                >
                  <X size={14} />
                </button>
              </span>
            {/each}

            {#if addingNewTag}
              <div class="flex items-center gap-1">
                <input
                  type="text"
                  bind:value={tag}
                  class="w-40 px-3 py-2 text-sm bg-[var(--color-ctp-base)] border-0 rounded-lg text-[var(--color-ctp-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-ctp-mauve)] transition-all placeholder-[var(--color-ctp-overlay0)] shadow-sm"
                  placeholder="New tag"
                  onkeydown={addTag}
                />
                <button
                  type="button"
                  onclick={addTag}
                  class="p-2 rounded-full text-[var(--color-ctp-mauve)] hover:bg-[var(--color-ctp-mauve)]/10 transition-all"
                >
                  <Plus size={16} />
                </button>
              </div>
            {:else}
              <button
                type="button"
                onclick={(e) => {
                  e.preventDefault();
                  addingNewTag = true;
                }}
                class="inline-flex items-center gap-1 py-1 px-3 text-sm rounded-full bg-transparent text-[var(--color-ctp-mauve)] border border-dashed border-[var(--color-ctp-mauve)]/50 hover:bg-[var(--color-ctp-mauve)]/10 transition-all"
              >
                <Plus size={14} />
                Add Tag
              </button>
            {/if}
          </div>
        </div>

        <!-- Reference -->
        <div></div>

        <!-- Footer -->
        <div
          class="flex justify-end gap-3 pt-6 mt-2 border-t border-[var(--color-ctp-surface0)]"
        >
          <button
            onclick={() => {
              editMode = !editMode;
            }}
            type="button"
            class="inline-flex items-center justify-center px-5 py-2.5 font-medium rounded-lg bg-transparent text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            class="inline-flex items-center justify-center gap-2 px-5 py-2.5 font-medium rounded-lg bg-gradient-to-r from-[var(--color-ctp-blue)] to-[var(--color-ctp-mauve)] text-[var(--color-ctp-crust)] hover:shadow-lg transition-all"
          >
            <Save size={18} />
            Update Experiment
          </button>
        </div>
      </form>
    </div>
    <!-- END FORM -->
  </div>
</div>
