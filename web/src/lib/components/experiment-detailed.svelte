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
    ChartLine,
    Eye,
    EyeClosed,
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";
  import ExperimentAiAnalysis from "./experiment-ai-analysis.svelte";

  let {
    experiment = $bindable(),
    selectedId = $bindable(),
    highlighted = $bindable(),
  }: {
    experiment: Experiment;
    selectedId: string | null;
    highlighted: string[];
  } = $props();

  let aiSuggestions = $state(null);
  let editMode = $state<boolean>(false);
</script>

{#if editMode}
  <EditExperimentModal bind:experiment bind:editMode />
{/if}

<article
  class="bg-[var(--color-ctp-base)] rounded-lg overflow-hidden shadow-lg"
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
        onclick={async () => {
          if (highlighted.at(-1) === experiment.id) {
            highlighted = [];
          } else {
            const response = await fetch(
              `/api/experiments/${experiment.id}/ref`,
            );
            const data = (await response.json()) as string[];
            highlighted = data;
          }
        }}
        class="p-1.5 text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] rounded-full transition-colors flex-shrink-0"
      >
        {#if highlighted.at(-1) === experiment.id}
          <EyeClosed size={16} />
        {:else}
          <Eye size={16} />
        {/if}
      </button>
      <button
        onclick={() => {
          if (selectedId === experiment.id) {
            selectedId = null;
          } else {
            selectedId = experiment.id;
          }
        }}
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
    <div>
      <div class="p-5 pb-0">
        <div class="flex items-center gap-2">
          <ChartLine size={16} class="text-[var(--color-ctp-mauve)]" />
          <h3 class="text-lg font-semibold text-[var(--color-ctp-mauve)]">
            Charts
          </h3>
        </div>
      </div>
      <InteractiveChart {experiment} />
    </div>
  {/if}

  <!-- AI Analysis section -->
  <!-- <ExperimentAiAnalysis {experiment} bind:aiSuggestions /> -->
</article>
