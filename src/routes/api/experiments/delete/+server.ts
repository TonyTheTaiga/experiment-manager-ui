import { deleteExperiment } from "$lib/server/database";
import { json } from "@sveltejs/kit";

export async function POST({ request }) {
  let data = await request.json();
  let id = data["id"];
  await deleteExperiment(id);
  return json({ success: true })
}
