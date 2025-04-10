import { json, error } from "@sveltejs/kit";
import { createReference, getReferenceChain } from "$lib/server/database";


export async function GET({
  params: { slug },
}: {
  params: { slug: string };
}) {
  const references = await getReferenceChain(slug);
  return json(references.map(experiment => experiment.id))
}


export async function POST({
  params: { slug },
  request,
}: {
  params: { slug: string };
  request: Request;
}) {
  const { referenceId } = await request.json();
  await createReference(slug, referenceId);
  return new Response(
    JSON.stringify({ message: "Reference created successfully" }),
    {
      status: 201,
      headers: { "Content-Type": "application/json" },
    },
  );
}
