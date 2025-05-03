import * as server from '../entries/pages/_page.server.ts.js';

export const index = 2;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_page.svelte.js')).default;
export { server };
export const server_id = "src/routes/+page.server.ts";
export const imports = ["_app/immutable/nodes/2.cjXnHzbr.js","_app/immutable/chunks/disclose-version.Ca4v9U1G.js","_app/immutable/chunks/runtime.nu9hB44V.js","_app/immutable/chunks/render.gNse54o8.js","_app/immutable/chunks/props.Ds-bfC-L.js","_app/immutable/chunks/entry.t6GxxK2p.js","_app/immutable/chunks/index-client.C4za7ih8.js","_app/immutable/chunks/legacy.DAIPZwv3.js","_app/immutable/chunks/lifecycle.CHHGwKqA.js","_app/immutable/chunks/slot.JJMjxmbe.js"];
export const stylesheets = ["_app/immutable/assets/2.l1tQnjQF.css"];
export const fonts = [];
