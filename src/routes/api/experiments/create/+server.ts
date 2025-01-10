import { createExperiment } from "$lib/server/database";
import { json } from "@sveltejs/kit";

export async function POST({ request }) {
  try {
    let data = await request.json();
    let name = data["name"];
    let description = data["description"];
    let hyperparams = data["hyperparams"];

    if (typeof hyperparams === "string") {
      try {
        hyperparams = JSON.parse(hyperparams);
      } catch (e) {
        console.error("Failed to parse hyperparams:", e);
        return json({ error: "Invalid hyperparams format" }, { status: 400 });
      }
    }

    const experiment = await createExperiment(name, description, hyperparams);
    return json({ success: true, experiment: experiment });
  } catch (error: unknown) {
    console.error("Error in POST handler:", error);
    return json(
      {
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 },
    );
  }
}
