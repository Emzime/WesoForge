<script lang="ts">
  import type { Snippet } from 'svelte';

  type Props = {
    title: string;
    onClose: () => void;
    ariaCloseLabel?: string;
    maxWidthClass?: string;
    bodyClass?: string;
    headerActions?: Snippet;
    children?: Snippet;
  };

  let { title, onClose, ariaCloseLabel, maxWidthClass = 'max-w-lg', bodyClass = '', headerActions, children }: Props = $props();
  const closeLabel = $derived(ariaCloseLabel ?? `Close ${title}`);
</script>

<div class="fixed inset-0 z-50 flex items-center justify-center p-6">
  <button
    type="button"
    class="absolute inset-0 bg-black/60"
    aria-label={closeLabel}
    onclick={onClose}
  ></button>

  <div class={`relative w-full ${maxWidthClass} flex max-h-[85vh] flex-col rounded border border-border bg-surface p-4 shadow-lg`}>
    <div class="flex items-center justify-between gap-3">
      <h2 class="text-sm font-semibold">{title}</h2>
      <div class="flex items-center gap-2">
        {@render headerActions?.()}
        <button
          type="button"
          class="rounded border border-border bg-bg px-2 py-2 text-fg transition-colors hover:border-accent/60 hover:text-accent"
          aria-label={closeLabel}
          onclick={onClose}
        >
          <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
          </svg>
        </button>
      </div>
    </div>

    <div class={`mt-4 min-h-0 flex-1 ${bodyClass}`}>
      {@render children?.()}
    </div>
  </div>
</div>
