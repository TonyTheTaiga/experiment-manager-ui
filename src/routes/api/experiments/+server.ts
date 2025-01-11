import { json, error } from "@sveltejs/kit";
import { getExperiments } from "$lib/server/database.js";

export async function GET() {
  try {
    const experiments = await getExperiments();
    return json(experiments);
  } catch (err) {
    console.log('error fetching experiments');

    if (err instanceof Error) {
      throw error(500, err.message);
    }

    throw error(500, 'Internal Error');

  }
}
