<script lang="ts">
    let { toggleIsOpen }: { toggleIsOpen: () => void } = $props();
</script>

<form method="POST" action="?/create" class="p-4">
    <div>
        <label class="block text-sm text-gray-700" for="name">Name</label>
        <input
            name="experiment-name"
            type="text"
            class="w-full px-3 py-2 border border-gray-300 rounded-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
    </div>
    <div>
        <label class="block text-sm text-gray-700" for="name">Description</label
        >
        <input
            name="experiment-description"
            type="text"
            class="w-full px-3 py-2 border border-gray-300 rounded-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
    </div>
    <div class="pt-4 border-t border-gray-200 flex gap-2 justify-end">
        <button
            type="submit"
            class="px-4 py-2 bg-gray-300 text-gray-700 rounded-sm hover:bg-gray-400 transition-colors"
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
