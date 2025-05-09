import { json } from "@sveltejs/kit";
import { getExperiment, updateExperiment, createReference } from "$lib/server/database";

export async function GET({ params: { slug } }: { params: { slug: string } }) {
  const experiment = await getExperiment(slug);
  return json(experiment);
}

export async function POST({
  params: { slug },
  request,
}: {
  params: { slug: string };
  request: Request;
}) {
  let data = await request.json();
  await updateExperiment(slug, {
    name: data.name,
    description: data.description,
    tags: data.tags,
  });

  return new Response(
    JSON.stringify({ message: "Experiment updated successfully" }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}
