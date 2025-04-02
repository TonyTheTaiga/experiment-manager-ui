import { json } from "@sveltejs/kit";
import { getExperiment, updateExperiment } from "$lib/server/database";

export async function GET(event) {
	const experiment = await getExperiment(event.params["slug"]);
	return json(experiment);
}

export async function POST({ params: { slug }, request }) {
	let data = await request.json();
	await updateExperiment(slug, {
		name: data.name,
		description: data.description,
	});
	return json({ success: true });
}
