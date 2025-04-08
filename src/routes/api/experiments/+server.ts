import { json, error } from "@sveltejs/kit";
import { getExperiments } from "$lib/server/database.js";

export async function GET({ url }: { url: URL }) {
  const name_filter = url.searchParams.get("startwith") || "";
  try {
    const experiments = await getExperiments(name_filter);
    return json(experiments);
  } catch (err) {
    if (err instanceof Error) {
      throw error(500, err.message);
    }

    throw error(500, "Internal Error");
  }
}
