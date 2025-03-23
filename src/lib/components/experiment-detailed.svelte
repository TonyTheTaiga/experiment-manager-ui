<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Minimize2, X, Clock, Tag, Settings } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import { marked } from "marked";

  let { experiment, toggleToggleId }: { experiment: Experiment, toggleToggleId: (id: string) => void } = $props();
  let aiAnalysis: string | null = $state(null);
</script>

<article class="rounded-lg overflow-hidden border border-ctp-surface1 bg-ctp-base shadow-lg">
  <!-- Header with actions -->
  <header class="bg-ctp-mantle border-b border-ctp-surface1 p-4 flex justify-between items-center">
    <h2 class="text-2xl font-medium text-ctp-text">
      {experiment.name}
    </h2>
    <div class="flex items-center gap-3">
      <button
        onclick={() => toggleToggleId(experiment.id)}
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface0 rounded-full transition-colors"
        aria-label="Minimize"
      >
        <Minimize2 size={16} />
      </button>
      <form method="POST" action="?/delete" class="flex items-center">
        <input type="hidden" name="id" value={experiment.id} />
        <button
          type="submit"
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0 rounded-full transition-colors"
          aria-label="Delete"
        >
          <X size={16} />
        </button>
      </form>
    </div>
  </header>

  <!-- Metadata section -->
  <div class="p-5 border-b border-ctp-surface0 bg-ctp-base">
    <div class="flex items-center gap-6 mb-4 text-sm text-ctp-subtext0">
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
              <span class="px-2 py-0.5 text-xs bg-ctp-surface0 text-ctp-lavender rounded-full">
                {tag}
              </span>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    {#if experiment.description}
      <p class="text-sm text-ctp-text leading-relaxed py-2 border-l-2 border-ctp-mauve pl-3 my-3 max-w-prose">
        {experiment.description}
      </p>
    {/if}
  </div>

  <!-- Parameters section -->
  {#if experiment.hyperparams}
    <div class="p-5 border-b border-ctp-surface0 bg-ctp-base">
      <div class="flex items-center gap-2 mb-3 text-ctp-lavender">
        <Settings size={16} />
        <h3 class="font-medium">Parameters</h3>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {#each experiment.hyperparams as param}
          <div class="flex items-center justify-between bg-ctp-mantle p-3 rounded-md">
            <span class="text-sm font-medium text-ctp-subtext1">{param.key}</span>
            <span class="text-sm text-ctp-text px-2 py-1 bg-ctp-surface0 rounded">{param.value}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Metrics section -->
  {#if experiment.availableMetrics}
    <div class="border-b border-ctp-surface0 bg-ctp-base">
      <div class="p-5 pb-0">
        <h3 class="text-lg font-medium text-ctp-mauve">Charts</h3>
      </div>
      <InteractiveChart {experiment} />
    </div>
  {/if}

  <!-- AI Analysis section -->
  <div class="bg-ctp-base">
    <div class="p-5 pb-0">
      <h3 class="text-lg font-medium text-ctp-blue">AI Analysis</h3>
    </div>
    <div class="p-5">
      {#if aiAnalysis}
        <div class="markdown-preview rounded-md overflow-hidden border border-ctp-surface1 shadow-inner">{@html marked(aiAnalysis)}</div>
      {:else}
        <div class="flex flex-col items-center justify-center p-8 bg-ctp-mantle rounded-md">
          <p class="text-ctp-subtext0 text-sm mb-4">No analysis available yet</p>
          <button
            class="px-4 py-2 bg-ctp-blue/90 hover:bg-ctp-blue text-ctp-mantle rounded-md transition-colors duration-200 font-medium flex items-center gap-2 shadow-sm"
            onclick={async () => {
              console.log("AI Analysis triggered");
              let response = await fetch(`/api/experiments/${experiment.id}/analysis`);
              let { analysis } = await response.json();
              aiAnalysis = analysis;
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
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
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
    color: var(--color-ctp-mauve);
    letter-spacing: -0.02em;
  }

  .markdown-preview :global(h2) {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--color-ctp-surface0);
    color: var(--color-ctp-lavender);
    letter-spacing: -0.01em;
  }

  .markdown-preview :global(h3) {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1.3rem;
    margin-bottom: 0.6rem;
    color: var(--color-ctp-blue);
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
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 85%;
    border: 1px solid var(--color-ctp-surface0);
  }

  .markdown-preview :global(code) {
    background-color: rgba(203, 166, 247, 0.15);
    border-radius: 3px;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
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
    border-left: 0.25rem solid var(--color-ctp-mauve);
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
    color: var(--color-ctp-blue);
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