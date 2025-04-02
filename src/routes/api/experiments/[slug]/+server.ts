import { json } from "@sveltejs/kit";
import { getExperiment } from "$lib/server/database";

export async function GET(event) {
	const experiment = await getExperiment(event.params["slug"]);
	return json(experiment.data);
}
