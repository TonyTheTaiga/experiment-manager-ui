
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```bash
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const XPC_FLAGS: string;
	export const __CFBundleIdentifier: string;
	export const VIMRUNTIME: string;
	export const XDG_DATA_DIRS: string;
	export const ZELLIJ_PANE_ID: string;
	export const XPC_SERVICE_NAME: string;
	export const STARSHIP_SHELL: string;
	export const MANPATH: string;
	export const PATH: string;
	export const HOMEBREW_CELLAR: string;
	export const INFOPATH: string;
	export const PWD: string;
	export const GHOSTTY_RESOURCES_DIR: string;
	export const SHELL: string;
	export const HOMEBREW_PREFIX: string;
	export const OPENAI_API_KEY: string;
	export const ZELLIJ: string;
	export const MODULAR_HOME: string;
	export const _: string;
	export const LOGNAME: string;
	export const ZELLIJ_SESSION_NAME: string;
	export const TERM_PROGRAM: string;
	export const NVIM: string;
	export const HOMEBREW_REPOSITORY: string;
	export const GOOGLE_APPLICATION_CREDENTIALS: string;
	export const TERM: string;
	export const GHOSTTY_BIN_DIR: string;
	export const TMPDIR: string;
	export const COMMAND_MODE: string;
	export const LANG: string;
	export const MASON: string;
	export const LDFLAGS: string;
	export const NVM_CD_FLAGS: string;
	export const HOME: string;
	export const NVM_DIR: string;
	export const VIM: string;
	export const TERM_PROGRAM_VERSION: string;
	export const SSH_AUTH_SOCK: string;
	export const GHOSTTY_SHELL_INTEGRATION_NO_SUDO: string;
	export const USER: string;
	export const COLORTERM: string;
	export const CPPFLAGS: string;
	export const OLDPWD: string;
	export const STARSHIP_SESSION_KEY: string;
	export const VIRTUAL_ENV: string;
	export const NVIM_LOG_FILE: string;
	export const PNPM_HOME: string;
	export const SHLVL: string;
	export const __CF_USER_TEXT_ENCODING: string;
	export const MYVIMRC: string;
	export const NVM_INC: string;
	export const NVM_BIN: string;
	export const TERMINFO: string;
	export const VIRTUAL_ENV_PROMPT: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	export const PUBLIC_SUPABASE_URL: string;
	export const PUBLIC_SUPABASE_ANON_KEY: string;
	export const PUBLIC_ANTHROPIC_KEY: string;
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		XPC_FLAGS: string;
		__CFBundleIdentifier: string;
		VIMRUNTIME: string;
		XDG_DATA_DIRS: string;
		ZELLIJ_PANE_ID: string;
		XPC_SERVICE_NAME: string;
		STARSHIP_SHELL: string;
		MANPATH: string;
		PATH: string;
		HOMEBREW_CELLAR: string;
		INFOPATH: string;
		PWD: string;
		GHOSTTY_RESOURCES_DIR: string;
		SHELL: string;
		HOMEBREW_PREFIX: string;
		OPENAI_API_KEY: string;
		ZELLIJ: string;
		MODULAR_HOME: string;
		_: string;
		LOGNAME: string;
		ZELLIJ_SESSION_NAME: string;
		TERM_PROGRAM: string;
		NVIM: string;
		HOMEBREW_REPOSITORY: string;
		GOOGLE_APPLICATION_CREDENTIALS: string;
		TERM: string;
		GHOSTTY_BIN_DIR: string;
		TMPDIR: string;
		COMMAND_MODE: string;
		LANG: string;
		MASON: string;
		LDFLAGS: string;
		NVM_CD_FLAGS: string;
		HOME: string;
		NVM_DIR: string;
		VIM: string;
		TERM_PROGRAM_VERSION: string;
		SSH_AUTH_SOCK: string;
		GHOSTTY_SHELL_INTEGRATION_NO_SUDO: string;
		USER: string;
		COLORTERM: string;
		CPPFLAGS: string;
		OLDPWD: string;
		STARSHIP_SESSION_KEY: string;
		VIRTUAL_ENV: string;
		NVIM_LOG_FILE: string;
		PNPM_HOME: string;
		SHLVL: string;
		__CF_USER_TEXT_ENCODING: string;
		MYVIMRC: string;
		NVM_INC: string;
		NVM_BIN: string;
		TERMINFO: string;
		VIRTUAL_ENV_PROMPT: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		PUBLIC_SUPABASE_URL: string;
		PUBLIC_SUPABASE_ANON_KEY: string;
		PUBLIC_ANTHROPIC_KEY: string;
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
