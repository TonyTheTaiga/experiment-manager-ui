<div
  class={`
                    bg-white rounded-sm p-4 
                    ${
                      targetId === experiment.id
                        ? "md:col-span-2 lg:col-span-4 row-span-2 order-first"
                        : "order-none"
                    }
                `}
>
  <article class={"flex flex-col gap-1"}>
    {#if targetId !== experiment.id}
      <ExperimentSimple {experiment} {toggleToggleId} />
    {:else}
      <ExperimentDetailed {experiment} {toggleToggleId} />
    {/if}
  </article>
</div>
