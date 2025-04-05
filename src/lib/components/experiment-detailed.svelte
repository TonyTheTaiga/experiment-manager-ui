<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    Minimize2,
    X,
    Clock,
    Tag,
    Settings,
    Pencil,
    Info,
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";
  import { marked } from "marked";

  let {
    experiment = $bindable(),
    toggleToggleId,
  }: { experiment: Experiment; toggleToggleId: (id: string) => void } =
    $props();

  let aiSuggestions = $state(null);
  let aiAnalysis = $state<string | null>(null);
  let editMode = $state<boolean>(false);

  function toggleEditMode() {
    editMode = !editMode;
  }
</script>

{#if editMode}
  <EditExperimentModal bind:experiment {toggleEditMode} />
{/if}

<article
  class="bg-[var(--color-ctp-base)] border border-[var(--color-ctp-surface1)] rounded-lg overflow-hidden shadow-lg"
>
  <!-- Header with actions -->
  <header
    class="p-4 bg-[var(--color-ctp-mantle)] border-b border-[var(--color-ctp-surface0)] flex justify-between items-center"
  >
    <h2 class="text-xl font-semibold text-[var(--color-ctp-text)]">
      {experiment.name}
    </h2>
    <div class="flex items-center gap-3">
      <button
        onclick={() => {
          editMode = true;
        }}
        class="p-1.5 rounded-full text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] transition-colors"
      >
        <Pencil size={16} />
      </button>
      <button
        onclick={() => toggleToggleId(experiment.id)}
        class="p-1.5 rounded-full text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] transition-colors"
        aria-label="Minimize"
      >
        <Minimize2 size={16} />
      </button>
      <form method="POST" action="?/delete" class="flex items-center">
        <input type="hidden" name="id" value={experiment.id} />
        <button
          type="submit"
          class="p-1.5 rounded-full text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-red)] hover:bg-[var(--color-ctp-surface0)] transition-colors"
          aria-label="Delete"
        >
          <X size={16} />
        </button>
      </form>
    </div>
  </header>

  <!-- Metadata section -->
  <div class="p-5 border-b border-[var(--color-ctp-surface0)]">
    <div
      class="flex items-center gap-6 mb-4 text-[var(--color-ctp-subtext0)] text-sm"
    >
      <div class="flex items-center gap-1.5">
        <Clock size={14} />
        <time>
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "numeric",
          })}
        </time>
      </div>

      {#if experiment.tags && experiment.tags.length > 0}
        <div class="flex items-center gap-1.5">
          <Tag size={14} />
          <div class="flex flex-wrap gap-2">
            {#each experiment.tags as tag}
              <span
                class="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-[var(--color-ctp-surface0)] text-[var(--color-ctp-mauve)]"
              >
                {tag}
              </span>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    {#if experiment.description}
      <p
        class="text-[var(--color-ctp-text)] text-sm py-2 border-l-2 border-[var(--color-ctp-mauve)] pl-3 my-3 max-w-prose leading-relaxed"
      >
        {experiment.description}
      </p>
    {/if}
  </div>

  <!-- Parameters section -->
  {#if experiment.hyperparams}
    <div class="p-5 border-b border-[var(--color-ctp-surface0)]">
      <div class="flex items-center gap-2 mb-4">
        <Settings size={16} class="text-[var(--color-ctp-mauve)]" />
        <h3 class="text-lg font-semibold text-[var(--color-ctp-mauve)]">
          Parameters
        </h3>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {#each experiment.hyperparams as param}
          <div
            class="flex items-center bg-[var(--color-ctp-mantle)] p-3 rounded-md"
          >
            <span class="text-sm font-medium text-[var(--color-ctp-subtext1)]"
              >{param.key}</span
            >
            <div class="flex-grow"></div>
            <span
              class="text-sm text-[var(--color-ctp-text)] px-2 py-1 bg-[var(--color-ctp-surface0)] rounded"
              >{param.value}</span
            >
            <div class="flex items-center ml-2 relative group">
              {#if aiSuggestions && aiSuggestions[param.key]}
                <Info
                  size={16}
                  class="text-[var(--color-ctp-mauve)] cursor-pointer hover:text-pink-400 transition-colors"
                />
                <div
                  class="absolute bottom-full right-0 mb-2 w-60 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10"
                >
                  <div
                    class="bg-[var(--color-ctp-base)] text-[var(--color-ctp-text)] text-xs rounded-lg p-3 shadow-lg border border-[var(--color-ctp-surface1)]"
                  >
                    {aiSuggestions[param.key]}
                  </div>
                </div>
              {:else}
                <Info size={16} class="text-gray-400" />
              {/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Metrics section -->
  {#if experiment.availableMetrics}
    <div class="border-b border-[var(--color-ctp-surface0)]">
      <div class="p-5 pb-0">
        <h3 class="text-lg font-semibold text-[var(--color-ctp-mauve)]">
          Charts
        </h3>
      </div>
      <InteractiveChart {experiment} />
    </div>
  {/if}

  <!-- AI Analysis section -->
  <div>
    <div class="p-5 pb-0">
      <h3 class="text-lg font-semibold text-[var(--color-ctp-mauve)]">
        AI Analysis
      </h3>
    </div>
    <div class="p-5">
      {#if aiAnalysis}
        <div
          class="markdown-preview rounded-md overflow-hidden border border-[var(--color-ctp-surface1)] shadow-inner"
        >
          {@html marked(aiAnalysis)}
        </div>
      {:else}
        <div
          class="flex flex-col items-center justify-center p-8 bg-[var(--color-ctp-mantle)] rounded-md"
        >
          <p class="text-[var(--color-ctp-subtext0)] text-sm mb-4">
            No analysis available yet
          </p>
          <button
            class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-md bg-[var(--color-ctp-mauve)] text-[var(--color-ctp-crust)] hover:bg-[var(--color-ctp-lavender)] transition-colors"
            onclick={async () => {
              console.log("AI Analysis triggered");

              try {
                const results = await Promise.allSettled([
                  fetch(`/api/experiments/${experiment.id}/analysis`).then(
                    (res) => res.json(),
                  ),
                  fetch(
                    `/api/experiments/${experiment.id}/analysis/structured`,
                  ).then((res) => res.json()),
                ]);

                // Process results independently
                if (results[0].status === "fulfilled") {
                  aiAnalysis = results[0].value.analysis;
                } else {
                  console.error("Error fetching analysis:", results[0].reason);
                }

                if (results[1].status === "fulfilled") {
                  aiSuggestions = results[1].value;
                } else {
                  console.error(
                    "Error fetching hyperparameter suggestions:",
                    results[1].reason,
                  );
                }
              } catch (error) {
                console.error("Unexpected error during API calls:", error);
              }
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              ><path
                d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"
              /><path d="M9 18h6" /><path d="M10 22h4" /></svg
            >
            Analyze Experiment
          </button>
        </div>
      {/if}
    </div>
  </div>
</article>

<style>
  .markdown-preview {
    min-height: 300px;
    max-height: 70vh;
    padding: 2rem;
    box-sizing: border-box;
    display: block;
    width: 100%;
    font-family:
      system-ui,
      -apple-system,
      BlinkMacSystemFont,
      "Segoe UI",
      Roboto,
      sans-serif;
    line-height: 1.6;
    color: var(--color-ctp-text);
    overflow-y: auto;
    background-color: var(--color-ctp-mantle);
  }

  .markdown-preview :global(h1) {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--color-ctp-surface0);
    color: var(--color-ctp-sapphire);
    letter-spacing: -0.02em;
  }

  .markdown-preview :global(h2) {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--color-ctp-surface0);
    color: var(--color-ctp-sapphire);
    letter-spacing: -0.01em;
  }

  .markdown-preview :global(h3) {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1.3rem;
    margin-bottom: 0.6rem;
    color: var(--color-ctp-sapphire);
  }

  .markdown-preview :global(h4) {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
    color: var(--color-ctp-sapphire);
  }

  .markdown-preview :global(p) {
    margin-bottom: 1rem;
  }

  .markdown-preview :global(ul),
  .markdown-preview :global(ol) {
    margin-left: 2rem;
    margin-bottom: 1rem;
  }

  .markdown-preview :global(ul) {
    list-style-type: disc;
  }

  .markdown-preview :global(ol) {
    list-style-type: decimal;
  }

  .markdown-preview :global(li) {
    margin-bottom: 0.25rem;
  }

  .markdown-preview :global(li p) {
    margin-bottom: 0.5rem;
  }

  .markdown-preview :global(table) {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 1.5rem;
  }

  .markdown-preview :global(th),
  .markdown-preview :global(td) {
    border: 1px solid var(--color-ctp-surface0);
    padding: 0.5rem 0.75rem;
  }

  .markdown-preview :global(th) {
    background-color: var(--color-ctp-surface0);
    font-weight: 600;
    text-align: left;
    color: var(--color-ctp-lavender);
  }

  .markdown-preview :global(tr:nth-child(2n)) {
    background-color: var(--color-ctp-crust);
  }

  .markdown-preview :global(pre) {
    background-color: var(--color-ctp-crust);
    border-radius: 3px;
    padding: 1rem;
    overflow-x: auto;
    margin-bottom: 1rem;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 85%;
    border: 1px solid var(--color-ctp-surface0);
  }

  .markdown-preview :global(code) {
    background-color: rgba(203, 166, 247, 0.15);
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 85%;
    padding: 0.2em 0.4em;
    margin: 0;
    color: var(--color-ctp-pink);
  }

  .markdown-preview :global(pre code) {
    background-color: transparent;
    padding: 0;
    margin: 0;
    font-size: 100%;
  }

  .markdown-preview :global(blockquote) {
    margin-left: 0;
    padding: 1rem;
    color: var(--color-ctp-subtext0);
    border-left: 0.25rem solid var(--color-ctp-sapphire);
    margin-bottom: 1rem;
    background-color: var(--color-ctp-mantle);
    border-radius: 0 6px 6px 0;
  }

  .markdown-preview :global(hr) {
    height: 0.25rem;
    padding: 0;
    margin: 1.5rem 0;
    background-color: var(--color-ctp-surface0);
    border: 0;
  }

  .markdown-preview :global(a) {
    color: var(--color-ctp-sapphire);
    text-decoration: none;
  }

  .markdown-preview :global(a:hover) {
    text-decoration: underline;
  }

  .markdown-preview :global(img) {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1rem 0;
  }
</style>
