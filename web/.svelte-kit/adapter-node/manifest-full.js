export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["favicon.png"]),
	mimeTypes: {".png":"image/png"},
	_: {
		client: {"start":"_app/immutable/entry/start.Em5d6j7d.js","app":"_app/immutable/entry/app.CO_B8la8.js","imports":["_app/immutable/entry/start.Em5d6j7d.js","_app/immutable/chunks/entry.t6GxxK2p.js","_app/immutable/chunks/runtime.nu9hB44V.js","_app/immutable/chunks/index-client.C4za7ih8.js","_app/immutable/entry/app.CO_B8la8.js","_app/immutable/chunks/runtime.nu9hB44V.js","_app/immutable/chunks/render.gNse54o8.js","_app/immutable/chunks/disclose-version.Ca4v9U1G.js","_app/immutable/chunks/props.Ds-bfC-L.js","_app/immutable/chunks/index-client.C4za7ih8.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
