import { json } from "@sveltejs/kit";
import { getExperiments } from "$lib/server/database.js";

export async function GET(event) {
  const experiments = await getExperiments();
  return json(experiments);
}
