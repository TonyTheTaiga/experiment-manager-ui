import type { Actions } from './$types';
import type { PageServerLoad } from './$types';
import { createExperiment, getExperiments, deleteExeriment } from '$lib/server/database';
import { fail, redirect } from '@sveltejs/kit';


// Dummy Data
export const load: PageServerLoad = async ({ params }) => {
	let experiments = await getExperiments();
	return {
		experiments: experiments
	};
};

export const actions = {
	create: async ({ request }) => {
		const data = await request.formData();
		let name = data.get('experiment-name')?.toString();
		let description = data.get('experiment-description')?.toString();
		if (name && description) {
			console.log('creating new experiment...');
			await createExperiment(name, description);
		}

		redirect(303, '/');
	},
	delete: async ({ request }) => {
		const data = await request.formData();
		const id = Number(data.get('id'));
		try {
			console.log('deleteing experiment...');
			await deleteExeriment(id);
			return { success: true };
		} catch (error) {
			return fail(500, { message: 'Failed to delete experiment' });
		}
	}

} satisfies Actions;