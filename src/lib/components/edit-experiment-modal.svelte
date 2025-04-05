<script lang="ts">
	import { X, Save, TagIcon, Plus } from "lucide-svelte";
	import { enhance } from "$app/forms";

	let { experiment = $bindable(), toggleEditMode } = $props();

	let addingNewTag = $state(false);
	let tag = $state<string | null>(null);

	function addTag() {
		if (!tag) return;
		experiment.tags.push(tag);
		tag = null;
	}
</script>

<div
	class="fixed inset-0 bg-[var(--color-ctp-crust)]/60 backdrop-blur-sm
         flex items-center justify-center p-4 z-50"
>
	<!-- HEADER -->
	<div
		class="bg-[var(--color-ctp-base)] w-full max-w-xl rounded-lg border border-[var(--color-ctp-surface1)] shadow-lg overflow-hidden"
	>
		<div
			class="px-6 py-4 border-b border-[var(--color-ctp-surface0)] flex justify-between items-center"
		>
			<h2 class="text-xl font-medium text-[var(--color-ctp-text)]">
				Edit Experiment
			</h2>
			<button
				onclick={toggleEditMode}
				class="p-1.5 text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-text)] hover:bg-[var(--color-ctp-surface0)] rounded-full transition-colors"
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
				class="flex flex-col gap-6"
				use:enhance={({ formElement, formData, action, cancel, submitter }) => {
					experiment.name = formData.get("experiment-name");
					experiment.description = formData.get("experiment-description");
					return async ({ result, update }) => {
						toggleEditMode();
					};
				}}
			>
				<input
					class="hidden"
					id="experiment-id"
					name="experiment-id"
					value={experiment.id}
				/>
				<div class="space-y-2">
					<label
						class="text-sm font-medium text-[var(--color-ctp-text)]"
						for="name">Experiment Name</label
					>
					<input
						id="experiment-name"
						name="experiment-name"
						type="text"
						class="w-full px-3.5 py-2.5 bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors"
						placeholder="Enter experiment name"
						value={experiment.name}
					/>
				</div>

				<div class="space-y-2">
					<label
						class="text-sm font-medium text-[var(--color-ctp-text)]"
						for="description"
					>
						Description
					</label>
					<textarea
						id="experiment-description"
						name="experiment-description"
						rows="3"
						class="w-full px-3.5 py-2.5 bg-[var(--color-ctp-mantle)] border border-[var(--color-ctp-surface0)] rounded-md text-[var(--color-ctp-text)] focus:outline-none focus:border-[var(--color-ctp-mauve)] focus:ring-1 focus:ring-[var(--color-ctp-mauve)] transition-colors resize-none"
						placeholder="Briefly describe this experiment"
						value={experiment.description}
					></textarea>
				</div>

				<div class="space-y-3">
					<div class="flex items-center gap-2">
						<TagIcon size={16} class="text-[var(--color-ctp-mauve)]" />
						<h3 class="text-lg font-semibold text-[var(--color-ctp-blue)]">
							Tags
						</h3>
					</div>

					<div class="flex flex-wrap items-center gap-2">
						{#each experiment.tags as tag, i}
							<input type="hidden" value={tag} name="tags.{i}" />
							<span
								class="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-[var(--color-ctp-surface0)] text-[var(--color-ctp-mauve)] border border-[var(--color-ctp-surface0)] group"
							>
								{tag}
								<button
									type="button"
									class="text-[var(--color-ctp-subtext0)] hover:text-[var(--color-ctp-red)] transition-colors ml-1.5"
									onclick={() => experiment.tags.splice(i, 1)}
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

				<div
					class="flex justify-end gap-3 pt-4 mt-2 border-t border-[var(--color-ctp-surface0)]"
				>
					<button
						onclick={toggleEditMode}
						type="button"
						class="inline-flex items-center justify-center px-4 py-2 font-medium rounded-md bg-[var(--color-ctp-surface0)] text-[var(--color-ctp-text)] border border-[var(--color-ctp-surface1)] hover:bg-[var(--color-ctp-surface1)] transition-colors"
					>
						Cancel
					</button>
					<button
						type="submit"
						class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-md bg-[var(--color-ctp-mauve)] text-[var(--color-ctp-crust)] hover:bg-[var(--color-ctp-lavender)] transition-colors"
					>
						<Save size={16} />
						Update Experiment
					</button>
				</div>
			</form>
		</div>
		<!-- END FORM -->
	</div>
</div>
