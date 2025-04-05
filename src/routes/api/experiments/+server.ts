import { json, error } from "@sveltejs/kit";
import { getExperiments } from "$lib/server/database.js";

export async function GET() {
  console.log("GET /api/experiments");
  try {
    const experiments = await getExperiments();
    console.log(experiments);
    return json(experiments);
  } catch (err) {
    if (err instanceof Error) {
      throw error(500, err.message);
    }

    throw error(500, "Internal Error");
  }
}
