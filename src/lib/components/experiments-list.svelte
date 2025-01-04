<script lang="ts">
    import type { Experiment } from "$lib/types";
    import { Maximize2, Minimize2, X } from "lucide-svelte";
    import InteractiveChart from "./interactive-chart.svelte";

    let { experiments }: { experiments: Experiment[] } = $props();
    let expandedId = $state<number | null>(null);
    let chartComponent = $state<InteractiveChart | null>(null);

    function toggleExapandedId(id: number) {
        expandedId = expandedId === id ? null : id;
    }
</script>

<section>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {#each experiments as experiment}
            <div
                class={`
                    bg-white rounded-sm p-4 
                    ${
                        expandedId === experiment.id
                            ? "md:col-span-2 lg:col-span-4 row-span-2 order-first"
                            : "order-none"
                    }
                `}
            >
                <article class={"flex flex-col gap-1"}>
                    {#if expandedId !== experiment.id}
                        <!-- Condensed View -->
                        <div class="flex flex-row justify-between">
                            <h3 class="font-medium text-lg text-gray-900">
                                {experiment.name}
                            </h3>
                            <button
                                onclick={() => {
                                    toggleExapandedId(experiment.id);
                                }}
                            >
                                <Maximize2
                                    class="w-5 h-5 text-gray-400 hover:text-gray-600"
                                />
                            </button>
                        </div>

                        <p class="text-gray-400 text-sm leading-relaxed">
                            {experiment.description}
                        </p>
                        <div class="flex flex-row gap-1 text-sm text-gray-500">
                            <span>Groups:</span>
                            {#if experiment?.groups}
                                <ul class="flex flex-row gap-1">
                                    {#each experiment.groups as group}
                                        <li class="items-center">
                                            <span>{group}</span>
                                        </li>
                                    {/each}
                                </ul>
                            {/if}
                        </div>
                        <div class="text-sm">
                            <span class="text-gray-500">Status:</span>
                            <span
                                class={`
                                    ${
                                        experiment.jobState === 1
                                            ? "text-sky-400"
                                            : "text-orange-400"
                                    }
                                    `}
                                >{experiment.jobState === 1
                                    ? "Running"
                                    : "Stopped"}</span
                            >
                        </div>
                        <time class="text-gray-300 text-xs mt-auto pt-4">
                            Created: 00-00-0000: 00:00
                        </time>
                    {:else}
                        <!-- Expanded View -->
                        <div class="flex flex-row justify-between items-center">
                            <h3
                                class="font-medium text-lg text-gray-900 text-left"
                            >
                                {experiment.name}
                            </h3>
                            <div class="flex flex-row gap-2">
                                <button
                                    onclick={() => {
                                        toggleExapandedId(experiment.id);
                                    }}
                                >
                                    <Minimize2
                                        class="w-5 h-5 text-gray-400 hover:text-gray-600"
                                    />
                                </button>
                                <form
                                    class="w-5 h-5"
                                    method="POST"
                                    action="?/delete"
                                >
                                    <input
                                        type="hidden"
                                        name="id"
                                        value={experiment.id}
                                    />
                                    <button type="submit">
                                        <X
                                            class="text-gray-400 hover:text-gray-600"
                                        />
                                    </button>
                                </form>
                            </div>
                        </div>
                        <div class="text-gray-400 leading-relaxed">
                            <p>
                                {experiment.description}
                            </p>
                        </div>
                        <div class="flex flex-row gap-1 text-gray-500">
                            <span>Groups:</span>
                            {#if experiment?.groups}
                                <ul class="flex flex-row gap-1">
                                    {#each experiment.groups as group}
                                        <li class="items-center">
                                            <span>{group}</span>
                                        </li>
                                    {/each}
                                </ul>
                            {/if}
                        </div>
                        <!-- <InteractiveChart bind:this={chartComponent} /> -->
                    {/if}
                </article>
            </div>
        {/each}
    </div>
</section>
