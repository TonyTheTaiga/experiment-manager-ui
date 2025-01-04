import { createExperiment } from "$lib/server/database";
import { json } from "@sveltejs/kit";

export async function POST({ request }) {
  let data = await request.json();
  let name = data["name"];
  let description = data["description"];
  const experiment = await createExperiment(name, description);
  return json(experiment);
}
