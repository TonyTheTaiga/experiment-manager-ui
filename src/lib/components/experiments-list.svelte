<script lang="ts">
	import type { Experiment } from "$lib/types";
	import ExperimentCard from "./experiment-card.svelte";

	let { experiments = $bindable() }: { experiments: Experiment[] } = $props();

	let selectedId = $state<string | null>(null);
	function toggleToggleId(id: string) {
		if (selectedId === id) {
			selectedId = null;
		} else {
			selectedId = id;
		}
	}
</script>

<section>
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
		{#each experiments as experiment, idx (experiment.id)}
			<ExperimentCard
				bind:experiment={experiments[idx]}
				{toggleToggleId}
				targetId={selectedId}
			/>
		{/each}
	</div>

	{#if experiments.length === 0}
		<div
			class="flex flex-col items-center justify-center p-12 text-center bg-[var(--color-ctp-mantle)] rounded-lg border border-[var(--color-ctp-surface1)]"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				width="40"
				height="40"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="1.5"
				stroke-linecap="round"
				stroke-linejoin="round"
				class="text-[var(--color-ctp-overlay0)] mb-4"
				><path d="M5 3a2 2 0 0 0-2 2" /><path d="M19 3a2 2 0 0 1 2 2" /><path
					d="M21 19a2 2 0 0 1-2 2"
				/><path d="M5 21a2 2 0 0 1-2-2" /><path d="M9 3h1" /><path
					d="M9 21h1"
				/><path d="M14 3h1" /><path d="M14 21h1" /><path d="M3 9v1" /><path
					d="M21 9v1"
				/><path d="M3 14v1" /><path d="M21 14v1" /><path d="M8 12h.01" /><path
					d="M12 12h.01"
				/><path d="M16 12h.01" /></svg
			>
			<h3 class="text-lg font-medium text-[var(--color-ctp-text)] mb-2">
				No experiments yet
			</h3>
			<p class="text-[var(--color-ctp-subtext0)] max-w-md">
				Create your first experiment to start tracking metrics and see them
				displayed here.
			</p>
		</div>
	{/if}
</section>
