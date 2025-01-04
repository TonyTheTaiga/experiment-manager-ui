import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";
import { PUBLIC_SUPABASE_URL, PUBLIC_SUPABASE_ANON_KEY } from "$env/static/public"
import type { Database } from "./database.types";
import type { Experiment } from "$lib/types";

const supabaseUrl = PUBLIC_SUPABASE_URL;
const supabaseKey = PUBLIC_SUPABASE_ANON_KEY;

let client: SupabaseClient<Database>;

function getClient() {
    if (!client) {
        client = createClient<Database>(supabaseUrl, supabaseKey)
    }

    return client
}

export async function createExperiment(name: string, description: string) {
    client = getClient();
    const { error } = await client.from('experiment').insert({ name: name, description: description, job_state: 0 });
    if (error) {
        throw new Error("Failed to create experiement");
    }

}

export async function getExperiments() {
    client = getClient();
    const { data, error } = await client.from('experiment').select();
    if (error) {
        throw new Error("Failed to fetch experiments")
    }

    let experiments = data.map((query_data) => {
        return { id: query_data['id'], name: query_data['name'], description: query_data['description'], jobState: query_data['job_state'] }
    });
    console.log(experiments);

    return experiments


}

export async function deleteExeriment(id: number) {
    client = getClient();
    const response = await client.from('experiment').delete().eq('id', id);
    console.log(console);
}