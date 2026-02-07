<!-- Patch snippet for App.svelte: GPU display line -->
{#if snapshot?.gpus && snapshot.gpus.length > 0}
  <div class="mt-2 text-sm opacity-80">
    {#each snapshot.gpus as gpu}
      <div>
        GPU:
        {#if gpu.device_name}
          {gpu.device_name} ({gpu.device_key})
        {:else}
          {gpu.device_key}
        {/if}
        - {gpu.enabled ? 'enabled' : 'disabled'}
      </div>
    {/each}
  </div>
{/if}
