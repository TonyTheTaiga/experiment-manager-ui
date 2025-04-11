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
		client: {"start":"_app/immutable/entry/start.CW8uFo1v.js","app":"_app/immutable/entry/app.DMhNzXCd.js","imports":["_app/immutable/entry/start.CW8uFo1v.js","_app/immutable/chunks/entry.BqIYnHe0.js","_app/immutable/chunks/runtime.nu9hB44V.js","_app/immutable/chunks/index-client.C4za7ih8.js","_app/immutable/entry/app.DMhNzXCd.js","_app/immutable/chunks/runtime.nu9hB44V.js","_app/immutable/chunks/render.gNse54o8.js","_app/immutable/chunks/disclose-version.Ca4v9U1G.js","_app/immutable/chunks/props.Ds-bfC-L.js","_app/immutable/chunks/index-client.C4za7ih8.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('../output/server/nodes/0.js')),
			__memo(() => import('../output/server/nodes/1.js')),
			__memo(() => import('../output/server/nodes/2.js'))
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
