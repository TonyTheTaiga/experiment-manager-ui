<script lang="ts">
    import { enhance } from "$app/forms";

    let { toggleIsOpen }: { toggleIsOpen: () => void } = $props();
</script>

<form method="POST" action="?/create" use:enhance class="flex-1 p-4">
    <div class="space-y-2">
        <label class="block text-sm text-gray-700" for="name">Name</label>
        <input
            name="experiment-name"
            type="text"
            class="w-full px-3 py-2 border border-gray-300 rounded-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
    </div>
    <div class="p-4 border-t border-gray-200 flex justify-end space-x-2">
        <button
            type="submit"
            class="px-4 py-2 bg-blue-500 text-white rounded-sm hover:bg-blue-600 transition-colors"
        >
            Submit
        </button>
        <button
            onclick={toggleIsOpen}
            class="px-4 py-2 bg-gray-100 text-gray-700 rounded-sm hover:bg-gray-200 transition-colors"
        >
            Close
        </button>
    </div>
</form>
