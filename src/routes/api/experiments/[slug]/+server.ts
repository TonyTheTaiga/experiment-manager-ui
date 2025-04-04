import { getExperiment, updateExperiment } from '$lib/server/database';
import { json } from '@sveltejs/kit';

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
  const data = await request.json();
  await updateExperiment(slug, {
    name: data.name,
    description: data.description,
    tags: data.tags,
  });
  return json({ success: true });
}
