<script lang="ts">
  import { marked } from "marked";
  import type { Experiment } from "$lib/types";

  let {
    experiment = $bindable(),
    aiSuggestions = $bindable(),
  }: {
    experiment: Experiment;
    aiSuggestions: any;
  } = $props();

  let aiAnalysis = $state<string | null>(null);

  async function triggerAnalysis() {
    try {
      const results = await Promise.allSettled([
        fetch(`/api/ai/analysis?experimentId=${experiment.id}`).then((res) =>
          res.json(),
        ),
        // Keeping this as structured endpoint for now
        fetch(`/api/ai/analysis/structured?experimentId=${experiment.id}`).then(
          (res) => res.json(),
        ),
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
  }
</script>

<div>
  <div class="p-5 pb-0">
    <h3 class="text-lg font-semibold text-ctp-mauve">AI Analysis</h3>
  </div>
  <div class="p-5">
    {#if aiAnalysis}
      <div
        class="markdown-preview rounded-md overflow-hidden border border-ctp-surface1 shadow-inner"
      >
        {@html marked(aiAnalysis)}
      </div>
    {:else}
      <div
        class="flex flex-col items-center justify-center p-8 bg-ctp-mantle rounded-md"
      >
        <p class="text-ctp-subtext0 text-sm mb-4">No analysis available yet</p>
        <button
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-md bg-ctp-mauve text-ctp-crust hover:bg-ctp-lavender transition-colors"
          onclick={triggerAnalysis}
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
