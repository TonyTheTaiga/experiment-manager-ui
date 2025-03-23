<script>
    import "../app.css";
</script>

<div class="h-screen bg-ctp-base text-ctp-text">
    <slot />
</div>

<style>
    :global(.bg-ctp-base) {
        background-color: var(--color-ctp-base);
    }
    :global(.bg-ctp-mantle) {
        background-color: var(--color-ctp-mantle);
    }
    :global(.bg-ctp-crust) {
        background-color: var(--color-ctp-crust);
    }
    :global(.bg-ctp-surface0) {
        background-color: var(--color-ctp-surface0);
    }
    :global(.bg-ctp-surface1) {
        background-color: var(--color-ctp-surface1);
    }
    :global(.bg-ctp-surface2) {
        background-color: var(--color-ctp-surface2);
    }
    :global(.border-ctp) {
        border-color: var(--color-ctp-surface0);
    }
    :global(.border-ctp-surface0) {
        border-color: var(--color-ctp-surface0);
    }
    :global(.border-ctp-surface1) {
        border-color: var(--color-ctp-surface1);
    }
    :global(.text-ctp) {
        color: var(--color-ctp-text);
    }
    :global(.text-ctp-text) {
        color: var(--color-ctp-text);
    }
    :global(.text-ctp-subtext0) {
        color: var(--color-ctp-subtext0);
    }
    :global(.text-ctp-subtext1) {
        color: var(--color-ctp-subtext1);
    }
    :global(.text-ctp-overlay0) {
        color: var(--color-ctp-overlay0);
    }
    :global(.bg-ctp-lavender) {
        background-color: var(--color-ctp-lavender);
    }
    :global(.bg-ctp-blue) {
        background-color: var(--color-ctp-blue);
    }
    :global(.bg-ctp-mauve) {
        background-color: var(--color-ctp-mauve);
    }
    :global(.bg-ctp-red) {
        background-color: var(--color-ctp-red);
    }
    :global(.text-ctp-lavender) {
        color: var(--color-ctp-lavender);
    }
    :global(.text-ctp-blue) {
        color: var(--color-ctp-blue);
    }
    :global(.text-ctp-mauve) {
        color: var(--color-ctp-mauve);
    }
    :global(.text-ctp-red) {
        color: var(--color-ctp-red);
    }
    :global(.text-ctp-rosewater) {
        color: var(--color-ctp-rosewater);
    }
    :global(.text-ctp-flamingo) {
        color: var(--color-ctp-flamingo);
    }
    :global(.text-ctp-pink) {
        color: var(--color-ctp-pink);
    }
    :global(.hover\:bg-ctp-surface0:hover) {
        background-color: var(--color-ctp-surface0);
    }
    :global(.hover\:bg-ctp-lavender:hover) {
        background-color: var(--color-ctp-lavender);
    }
    :global(.hover\:bg-ctp-blue:hover) {
        background-color: var(--color-ctp-blue);
    }
    :global(.hover\:bg-ctp-blue\/90:hover) {
        background-color: rgba(137, 180, 250, 0.9);
    }
    :global(.hover\:text-ctp-text:hover) {
        color: var(--color-ctp-text);
    }
    :global(.hover\:text-ctp-red:hover) {
        color: var(--color-ctp-red);
    }
    :global(.hover\:text-ctp-blue:hover) {
        color: var(--color-ctp-blue);
    }
    :global(.hover\:border-ctp-blue\/50:hover) {
        border-color: rgba(137, 180, 250, 0.5);
    }
    :global(.bg-ctp-blue\/90) {
        background-color: rgba(137, 180, 250, 0.9);
    }
    :global(.focus-visible\:ring-2:focus-visible) {
        --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
        --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(2px + var(--tw-ring-offset-width)) var(--tw-ring-color);
        box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
    }
    :global(.focus-visible\:ring-ctp-blue\/30:focus-visible) {
        --tw-ring-color: rgba(137, 180, 250, 0.3);
    }
</style>