<script lang="ts">
    import type { Experiment } from "$lib/types";
    import { Maximize2, Minimize2 } from "lucide-svelte";
    import InteractiveChart from "./interactive-chart.svelte";

    let { experiments }: { experiments: Experiment[] } = $props();
    let expandedId = $state<number | null>(null);

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
                <article class={"flex flex-col gap-2"}>
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
                                    class="w-5 h-5 text-gray-300 hover:text-gray-600"
                                />
                            </button>
                        </div>

                        <p class="text-gray-600 text-sm leading-relaxed">
                            Lorem ipsum odor amet, consectetuer adipiscing elit.
                        </p>
                        <div class="flex flex-row gap-2">
                            <span class="text-gray-600 text-sm">Groups:</span>
                            {#if experiment?.groups}
                                <ul class="flex flex-row gap-2">
                                    {#each experiment.groups as group}
                                        <li class="items-center">
                                            <span class="text-gray-500 text-sm"
                                                >{group}</span
                                            >
                                        </li>
                                    {/each}
                                </ul>
                            {/if}
                        </div>
                        <time class="text-gray-400 text-xs mt-auto pt-4">
                            Created: 00-00-0000: 00:00
                        </time>
                    {:else}
                        <!-- Expanded View -->
                        <div class="flex flex-row justify-between">
                            <h3
                                class="font-medium text-lg text-gray-900 text-left"
                            >
                                {experiment.name}
                            </h3>
                            <button
                                onclick={() => {
                                    toggleExapandedId(experiment.id);
                                }}
                            >
                                <Minimize2
                                    class="w-5 h-5 text-gray-300 hover:text-gray-600"
                                />
                            </button>
                        </div>

                        <p class="text-gray-400 leading-relaxed">
                            Lorem ipsum odor amet, consectetuer adipiscing elit.
                        </p>
                        <textarea></textarea>
                        <InteractiveChart />
                    {/if}
                </article>
            </div>
        {/each}
    </div>
</section>
