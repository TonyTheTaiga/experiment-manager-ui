import { deleteExeriment } from "$lib/server/database";
import { json } from "@sveltejs/kit";

export async function POST({ request }) {
  let data = await request.json();
  let id = data["id"];
  const experiment = await deleteExeriment(id);
  return json(experiment);
}
