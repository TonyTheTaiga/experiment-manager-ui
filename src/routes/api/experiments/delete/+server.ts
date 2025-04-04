import { deleteExperiment } from '$lib/server/database';
import { json } from '@sveltejs/kit';

export async function POST({ request }) {
  const data = await request.json();
  const id = data.id;
  await deleteExperiment(id);
  return json({ success: true });
}
