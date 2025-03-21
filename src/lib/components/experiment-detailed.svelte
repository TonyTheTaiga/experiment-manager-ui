<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Minimize2, X } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import { marked } from "marked";

  let { experiment, toggleToggleId }: { experiment: Experiment, toggleToggleId: (id: string) => void } = $props();
  let aiAnalysis: string | null = $state(null);
</script>

<article class="p-4 bg-white">
  <div class="flex justify-between items-center">
    <time class="text-sm text-gray-400">
      {new Date(experiment.createdAt).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "numeric",
      })}
    </time>
    <div class="flex items-center gap-3">
      <button
        onclick={() => toggleToggleId(experiment.id)}
        class="text-gray-600 hover:text-black transition-colors flex items-center justify-center"
      >
        <Minimize2 size={16} />
      </button>
      <form method="POST" action="?/delete" class="flex items-center">
        <input type="hidden" name="id" value={experiment.id} />
        <button
          type="submit"
          class="text-gray-600 hover:text-red-600 transition-colors flex items-center justify-center"
        >
          <X size={16} />
        </button>
      </form>
    </div>
  </div>

  <h2 class="text-2xl font-medium text-gray-900 mb-6">
    {experiment.name}
  </h2>

  <p class="text-sm text-gray-500 mb-2 leading-relaxed">
    {experiment.description}
  </p>

  {#if experiment.tags && experiment.tags.length > 0}
    <div class="flex flex-wrap items-center gap-2 mb-2">
      <span class="text-sm text-gray-600">Tags:</span>
      {#each experiment.tags as tag}
        <span class="px-2 py-1 text-xs bg-gray-50 text-gray-600 rounded-sm">
          {tag}
        </span>
      {/each}
    </div>
  {/if}

  {#if experiment.hyperparams}
    <div class="flex flex-wrap gap-4 mb-6">
      {#each experiment.hyperparams as param}
        <div class="flex items-center gap-1">
          <span class="text-xs text-gray-600">{param.key}</span>
          <span class="text-xs text-gray-400">{param.value}</span>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Metrics -->
  {#if experiment.availableMetrics}
    <div class="mb-6 rounded-md border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
      <h3 class="px-4 pt-4 text-lg">Charts</h3>
      <InteractiveChart {experiment} />
    </div>
  {/if}

  <!-- AI Analysis -->
  <div class="mb-6 rounded-md border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
    <h3 class="px-4 pt-4 text-lg">AI Analysis</h3>
    <div class='p-4'>
        {#if aiAnalysis}
            <div class="preview">{@html marked(aiAnalysis)}</div>
        {:else}
            <button
            class="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors duration-200"
            onclick={async () => {
                console.log("AI Analysis triggered");
                let response = await fetch(`/api/experiments/${experiment.id}/analysis`);
                let { analysis } = await response.json();
                aiAnalysis = analysis;
            }}
            >
                Analyze
            </button>
        {/if}
    </div>
  </div>
</article>

<style>
    .preview {
      height: 75%;
      padding: 2rem;
      box-sizing: border-box;
      display: block;
      width: 100%;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.5;
      color: #333;
      overflow-y: auto;
    }

    /* Use :global to style elements rendered from markdown */
    .preview :global(h1) {
      font-size: 2rem;
      font-weight: bold;
      margin-bottom: 1rem;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid #eaecef;
    }

    .preview :global(h2) {
      font-size: 1.5rem;
      font-weight: 600;
      margin-top: 1.5rem;
      margin-bottom: 0.8rem;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid #eaecef;
    }

    .preview :global(h3) {
      font-size: 1.25rem;
      font-weight: 600;
      margin-top: 1.3rem;
      margin-bottom: 0.6rem;
    }

    .preview :global(h4) {
      font-size: 1rem;
      font-weight: 600;
      margin-top: 1.2rem;
      margin-bottom: 0.5rem;
    }

    .preview :global(p) {
      margin-bottom: 1rem;
    }

    .preview :global(ul),
    .preview :global(ol) {
      margin-left: 2rem;
      margin-bottom: 1rem;
    }

    .preview :global(ul) {
      list-style-type: disc;
    }

    .preview :global(ol) {
      list-style-type: decimal;
    }

    .preview :global(li) {
      margin-bottom: 0.25rem;
    }

    .preview :global(li p) {
      margin-bottom: 0.5rem;
    }

    .preview :global(table) {
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 1.5rem;
    }

    .preview :global(th),
    .preview :global(td) {
      border: 1px solid #ddd;
      padding: 0.5rem 0.75rem;
    }

    .preview :global(th) {
      background-color: #f6f8fa;
      font-weight: 600;
      text-align: left;
    }

    .preview :global(tr:nth-child(2n)) {
      background-color: #f8f8f8;
    }

    .preview :global(pre) {
      background-color: #f6f8fa;
      border-radius: 3px;
      padding: 1rem;
      overflow-x: auto;
      margin-bottom: 1rem;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 85%;
    }

    .preview :global(code) {
      background-color: rgba(27, 31, 35, 0.05);
      border-radius: 3px;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 85%;
      padding: 0.2em 0.4em;
      margin: 0;
    }

    .preview :global(pre code) {
      background-color: transparent;
      padding: 0;
      margin: 0;
      font-size: 100%;
    }

    .preview :global(blockquote) {
      margin-left: 0;
      padding: 0 1rem;
      color: #6a737d;
      border-left: 0.25rem solid #dfe2e5;
      margin-bottom: 1rem;
    }

    .preview :global(hr) {
      height: 0.25rem;
      padding: 0;
      margin: 1.5rem 0;
      background-color: #e1e4e8;
      border: 0;
    }

    .preview :global(a) {
      color: #0366d6;
      text-decoration: none;
    }

    .preview :global(a:hover) {
      text-decoration: underline;
    }

    .preview :global(img) {
      max-width: 100%;
      height: auto;
      display: block;
      margin: 1rem 0;
    }
</style>
