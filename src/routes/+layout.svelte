<script>
    import "../app.css";
</script>

<div class="h-screen bg-ctp-base text-ctp-text">
    <slot />
</div>

<style>
    /* Catppuccin Mocha Theme System */
    
    /* Background Colors */
    :global(.bg-ctp-base) { background-color: var(--color-ctp-base); }
    :global(.bg-ctp-mantle) { background-color: var(--color-ctp-mantle); }
    :global(.bg-ctp-crust) { background-color: var(--color-ctp-crust); }
    :global(.bg-ctp-surface0) { background-color: var(--color-ctp-surface0); }
    :global(.bg-ctp-surface1) { background-color: var(--color-ctp-surface1); }
    :global(.bg-ctp-surface2) { background-color: var(--color-ctp-surface2); }
    
    /* Border Colors */
    :global(.border-ctp) { border-color: var(--color-ctp-surface0); }
    :global(.border-ctp-surface0) { border-color: var(--color-ctp-surface0); }
    :global(.border-ctp-surface1) { border-color: var(--color-ctp-surface1); }
    
    /* Text Colors */
    :global(.text-ctp) { color: var(--color-ctp-text); }
    :global(.text-ctp-text) { color: var(--color-ctp-text); }
    :global(.text-ctp-subtext0) { color: var(--color-ctp-subtext0); }
    :global(.text-ctp-subtext1) { color: var(--color-ctp-subtext1); }
    :global(.text-ctp-overlay0) { color: var(--color-ctp-overlay0); }
    
    /* Accent Background Colors */
    :global(.bg-ctp-mauve) { background-color: var(--color-ctp-mauve); }
    :global(.bg-ctp-lavender) { background-color: var(--color-ctp-lavender); }
    :global(.bg-ctp-blue) { background-color: var(--color-ctp-blue); }
    :global(.bg-ctp-sapphire) { background-color: var(--color-ctp-sapphire); }
    :global(.bg-ctp-sky) { background-color: var(--color-ctp-sky); }
    :global(.bg-ctp-red) { background-color: var(--color-ctp-red); }
    :global(.bg-ctp-peach) { background-color: var(--color-ctp-peach); }
    :global(.bg-ctp-yellow) { background-color: var(--color-ctp-yellow); }
    :global(.bg-ctp-green) { background-color: var(--color-ctp-green); }
    
    /* Accent Text Colors */
    :global(.text-ctp-mauve) { color: var(--color-ctp-mauve); }
    :global(.text-ctp-lavender) { color: var(--color-ctp-lavender); }
    :global(.text-ctp-blue) { color: var(--color-ctp-blue); }
    :global(.text-ctp-sapphire) { color: var(--color-ctp-sapphire); }
    :global(.text-ctp-sky) { color: var(--color-ctp-sky); }
    :global(.text-ctp-red) { color: var(--color-ctp-red); }
    :global(.text-ctp-peach) { color: var(--color-ctp-peach); }
    :global(.text-ctp-yellow) { color: var(--color-ctp-yellow); }
    :global(.text-ctp-green) { color: var(--color-ctp-green); }
    :global(.text-ctp-rosewater) { color: var(--color-ctp-rosewater); }
    :global(.text-ctp-flamingo) { color: var(--color-ctp-flamingo); }
    :global(.text-ctp-pink) { color: var(--color-ctp-pink); }
    
    /* Hover Colors */
    :global(.hover\:bg-ctp-surface0:hover) { background-color: var(--color-ctp-surface0); }
    :global(.hover\:bg-ctp-surface1:hover) { background-color: var(--color-ctp-surface1); }
    :global(.hover\:bg-ctp-surface2:hover) { background-color: var(--color-ctp-surface2); }
    :global(.hover\:bg-ctp-mauve:hover) { background-color: var(--color-ctp-mauve); }
    :global(.hover\:bg-ctp-lavender:hover) { background-color: var(--color-ctp-lavender); }
    :global(.hover\:bg-ctp-blue:hover) { background-color: var(--color-ctp-blue); }
    :global(.hover\:bg-ctp-sapphire:hover) { background-color: var(--color-ctp-sapphire); }
    :global(.hover\:bg-ctp-sky:hover) { background-color: var(--color-ctp-sky); }
    :global(.hover\:text-ctp-text:hover) { color: var(--color-ctp-text); }
    :global(.hover\:text-ctp-mauve:hover) { color: var(--color-ctp-mauve); }
    :global(.hover\:text-ctp-lavender:hover) { color: var(--color-ctp-lavender); }
    :global(.hover\:text-ctp-blue:hover) { color: var(--color-ctp-blue); }
    :global(.hover\:text-ctp-sapphire:hover) { color: var(--color-ctp-sapphire); }
    :global(.hover\:text-ctp-red:hover) { color: var(--color-ctp-red); }
    
    /* Opacity Variants */
    :global(.bg-ctp-mauve\/90) { background-color: rgba(203, 166, 247, 0.9); }
    :global(.bg-ctp-lavender\/90) { background-color: rgba(180, 190, 254, 0.9); }
    :global(.bg-ctp-blue\/90) { background-color: rgba(137, 180, 250, 0.9); }
    :global(.bg-ctp-sapphire\/90) { background-color: rgba(116, 199, 236, 0.9); }
    :global(.bg-ctp-blue\/80) { background-color: rgba(137, 180, 250, 0.8); }
    :global(.bg-ctp-blue\/20) { background-color: rgba(137, 180, 250, 0.2); }
    :global(.bg-ctp-blue\/10) { background-color: rgba(137, 180, 250, 0.1); }
    
    :global(.hover\:bg-ctp-mauve\/80:hover) { background-color: rgba(203, 166, 247, 0.8); }
    :global(.hover\:bg-ctp-blue\/80:hover) { background-color: rgba(137, 180, 250, 0.8); }
    :global(.hover\:bg-ctp-lavender\/80:hover) { background-color: rgba(180, 190, 254, 0.8); }
    
    :global(.hover\:border-ctp-mauve:hover) { border-color: var(--color-ctp-mauve); }
    :global(.hover\:border-ctp-blue:hover) { border-color: var(--color-ctp-blue); }
    :global(.hover\:border-ctp-mauve\/30:hover) { border-color: rgba(203, 166, 247, 0.3); }
    :global(.hover\:border-ctp-blue\/30:hover) { border-color: rgba(137, 180, 250, 0.3); }
    :global(.hover\:border-ctp-blue\/50:hover) { border-color: rgba(137, 180, 250, 0.5); }
    :global(.hover\:border-ctp-blue\/60:hover) { border-color: rgba(137, 180, 250, 0.6); }
    
    /* Focus Rings */
    :global(.focus-visible\:ring-2:focus-visible) {
        --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
        --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(2px + var(--tw-ring-offset-width)) var(--tw-ring-color);
        box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
    }
    :global(.focus-visible\:ring-ctp-mauve\/30:focus-visible) { --tw-ring-color: rgba(203, 166, 247, 0.3); }
    :global(.focus-visible\:ring-ctp-blue\/30:focus-visible) { --tw-ring-color: rgba(137, 180, 250, 0.3); }
    :global(.focus-visible\:ring-ctp-blue\/40:focus-visible) { --tw-ring-color: rgba(137, 180, 250, 0.4); }
    :global(.focus-visible\:outline-none:focus-visible) { outline: none; }
    
    /* Component System */
    
    /* Button System */
    :global(.ctp-btn) {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border-radius: 0.375rem;
        transition: all 150ms ease;
    }
    
    :global(.ctp-btn-primary) {
        background-color: var(--color-ctp-mauve);
        color: var(--color-ctp-crust);
    }
    :global(.ctp-btn-primary:hover) {
        background-color: var(--color-ctp-lavender);
    }
    :global(.ctp-btn-primary:focus-visible) {
        outline: none;
        box-shadow: 0 0 0 2px var(--color-ctp-base), 0 0 0 4px var(--color-ctp-mauve);
    }
    
    :global(.ctp-btn-secondary) {
        background-color: var(--color-ctp-surface0);
        color: var(--color-ctp-text);
        border: 1px solid var(--color-ctp-surface1);
    }
    :global(.ctp-btn-secondary:hover) {
        background-color: var(--color-ctp-surface1);
    }
    :global(.ctp-btn-secondary:focus-visible) {
        outline: none;
        box-shadow: 0 0 0 2px var(--color-ctp-base), 0 0 0 4px var(--color-ctp-lavender);
    }
    
    :global(.ctp-btn-outline) {
        background-color: transparent;
        color: var(--color-ctp-mauve);
        border: 1px solid var(--color-ctp-mauve);
    }
    :global(.ctp-btn-outline:hover) {
        background-color: rgba(203, 166, 247, 0.1);
    }
    :global(.ctp-btn-outline:focus-visible) {
        outline: none;
        box-shadow: 0 0 0 2px var(--color-ctp-base), 0 0 0 4px var(--color-ctp-mauve);
    }
    
    :global(.ctp-btn-danger) {
        background-color: var(--color-ctp-red);
        color: var(--color-ctp-crust);
    }
    :global(.ctp-btn-danger:hover) {
        background-color: var(--color-ctp-maroon);
    }
    :global(.ctp-btn-danger:focus-visible) {
        outline: none;
        box-shadow: 0 0 0 2px var(--color-ctp-base), 0 0 0 4px var(--color-ctp-red);
    }
    
    :global(.ctp-btn-sm) {
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
    }
    
    :global(.ctp-btn-xs) {
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
    }
    
    :global(.ctp-btn-icon) {
        padding: 0.375rem;
        border-radius: 9999px;
        color: var(--color-ctp-subtext0);
    }
    :global(.ctp-btn-icon:hover) {
        color: var(--color-ctp-text);
        background-color: var(--color-ctp-surface0);
    }
    
    /* Card System */
    :global(.ctp-card) {
        background-color: var(--color-ctp-base);
        border: 1px solid var(--color-ctp-surface1);
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    :global(.ctp-card-header) {
        padding: 1rem;
        background-color: var(--color-ctp-mantle);
        border-bottom: 1px solid var(--color-ctp-surface0);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    :global(.ctp-card-body) {
        padding: 1.25rem;
    }
    
    /* Tag System */
    :global(.ctp-tag) {
        display: inline-flex;
        align-items: center;
        padding: 0.125rem 0.5rem;
        font-size: 0.75rem;
        border-radius: 9999px;
        background-color: var(--color-ctp-surface0);
    }
    
    :global(.ctp-tag-lavender) { color: var(--color-ctp-lavender); }
    :global(.ctp-tag-blue) { color: var(--color-ctp-blue); }
    :global(.ctp-tag-mauve) { color: var(--color-ctp-mauve); }
    :global(.ctp-tag-sapphire) { color: var(--color-ctp-sapphire); }
    :global(.ctp-tag-green) { color: var(--color-ctp-green); }
    :global(.ctp-tag-peach) { color: var(--color-ctp-peach); }
    
    /* Typography System */
    :global(.ctp-heading-1) {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--color-ctp-text);
        line-height: 1.2;
    }
    
    :global(.ctp-heading-2) {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--color-ctp-mauve);
        line-height: 1.3;
    }
    
    :global(.ctp-heading-3) {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--color-ctp-blue);
        line-height: 1.4;
    }
    
    :global(.ctp-text) {
        color: var(--color-ctp-text);
        line-height: 1.5;
    }
    
    :global(.ctp-text-dim) {
        color: var(--color-ctp-subtext0);
        line-height: 1.5;
    }
    
    /* Form System */
    :global(.ctp-input) {
        width: 100%;
        padding: 0.625rem 0.875rem;
        background-color: var(--color-ctp-mantle);
        border: 1px solid var(--color-ctp-surface0);
        border-radius: 0.375rem;
        color: var(--color-ctp-text);
        transition: all 150ms ease;
    }
    :global(.ctp-input::placeholder) {
        color: var(--color-ctp-subtext0);
    }
    :global(.ctp-input:focus) {
        outline: none;
        border-color: var(--color-ctp-mauve);
        box-shadow: 0 0 0 1px var(--color-ctp-mauve);
    }
</style>