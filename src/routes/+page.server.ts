import type { Actions } from "./$types";
import type { PageServerLoad } from "./$types";
import { fail, redirect } from "@sveltejs/kit";

export const load: PageServerLoad = async ({ fetch }) => {
  const response = await fetch("/api/experiments");
  const data = await response.json();
  return {
    experiments: data,
  };
};

export const actions = {
  create: async ({ request, fetch }) => {
    const data = await request.formData();
    let name = data.get("experiment-name");
    let description = data.get("experiment-description");
    if (!name || !description) {
      return fail(400, { message: "Name and description are required" });
    }
    console.log("creating new experiment...");
    let response = await fetch("/api/experiments/create", {
      method: "POST",
      body: JSON.stringify({
        name: name,
        description: description,
      }),
    });
    if (!response.ok) {
      console.log(response);
      return fail(500, { message: "Failed to create experiment" });
    }
    redirect(303, "/");
  },

  delete: async ({ request, fetch }) => {
    const data = await request.formData();
    const id = Number(data.get("id"));
    if (!id) {
      return fail(400, { message: "ID is required" });
    }
    let response = await fetch("/api/experiments/delete", {
      method: "POST",
      body: JSON.stringify({
        id: id,
      }),
    });
    if (!response.ok) {
      console.log(response);
      return fail(500, { message: "Failed to delete experiment" });
    }
    redirect(303, "/");
  },
} satisfies Actions;
