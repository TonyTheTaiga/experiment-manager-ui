import type { Actions } from './$types';
import type { PageServerLoad } from './$types';


export const load: PageServerLoad = async ({ params }) => {
	return {
		experiments: [
			{ id: 1, name: 'Resnet', groups: ['Dogs', 'Animals'], running: false },
			{ id: 2, name: "DeezNuts3000", groups: ['Animals', 'Cats'], running: true },
			{ id: 3, name: "Hello, World", running: false },
			{ id: 4, name: "Top Secret", groups: ['CIA'], running: true }
		]
	};
};

export const actions = {
	create: async ({ cookies, request }) => {
		console.log('creating new experiment...');
		const data = await request.formData();
		console.log(data);
	}
} satisfies Actions;