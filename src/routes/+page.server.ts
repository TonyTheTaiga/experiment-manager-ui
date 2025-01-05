import type { Actions } from "./$types";
import type { PageServerLoad } from "./$types";
import { fail, redirect } from "@sveltejs/kit";
import type { HyperParam } from "$lib/types";

export const load: PageServerLoad = async ({ fetch }) => {
  const response = await fetch("/api/experiments");
  const data = await response.json();
  return {
    experiments: data,
  };
};

function parseFormData(formData: FormData) {
  const obj = Object.fromEntries(formData);
  const result: {
    hyperparams: HyperParam[];
    [key: string]: any;
  } = {
    hyperparams: [],
  };

  Object.entries(obj).forEach(([key, value]) => {
    if (key.startsWith("hyperparams.")) {
      const [_, index, field] = key.split(".");
      let idx = Number(index);
      if (!result.hyperparams[idx]) {
        result.hyperparams[idx] = { key: value as string, value: "" };
      } else {
        result.hyperparams[idx].value = value;
      }
    } else {
      result[key] = value;
    }
  });
  result.hyperparams = result.hyperparams.filter(Boolean);
  return result;
}

export const actions = {
  create: async ({ request, fetch }) => {
    const form = await request.formData();
    const data = parseFormData(form);
    let name = data["experiment-name"];
    let description = data["experiment-description"];
    let hyperparams = data["hyperparams"];
    if (!name || !description) {
      return fail(400, { message: "Name and description are required" });
    }
    console.log("creating new experiment...");
    let response = await fetch("/api/experiments/create", {
      method: "POST",
      body: JSON.stringify({
        name: name,
        description: description,
        hyperparams: hyperparams,
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
