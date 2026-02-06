WesoForge
├── UI/
|		├── public/
|		│	├── fonts/
|		│   |	└── itc-kabel-std/
|		│   |		└── ITCKabelStdMedium.TTF
|		│	|
|		│   └── logo-64.avif
|		│
|		├── src/
|		│   ├── components/
|		│   |	└── PopupFrame.svelte
|		│	|
|		│   ├── main.ts
|		│   ├── App.svelte
|		│   └── app.css
|		│
|		├── eslint.config.js
|		├── index.html
|		├── package.json
|		├── pnpm-lock.yaml
|		├── svelte.config.js
|		├── tsconfig.base.json
|		├── tsconfig.json
|		└── vite.config.ts
|
└── crates/
		├── chiavdf-fast/
		│   ├── native/
		│   |	└── chiavdf_fast_fallback.cpp
		│   |
		│   ├── src/
		│   |	├── api.rs
		│   |	├── ffi.rs
		│   |	└── lib.rs
		│   |
		|	├── build.rs
		│   └── Cargo.toml
		│
		├── client/
		│   ├── src
		│   |	├── bench.rs
		│   |	├── cli.rs
		│   |	├── constants.rs
		│   |	├── format.rs
		│   |	├── main.rs
		│   |	├── shutdown.rs
		│   |	├── terminal.rs
		│   |	└── ui.rs
		│   |
		│   ├── build.rs
		│   └── Cargo.toml
		│
		├── client-core/
		│   ├── src
		│   |	├── lib.rs
		│   |	└── submitter.rs
		│   |
		│   └── Cargo.toml
		│
		├── client-engine/
		│   ├── src
		│   |	├── api.rs
		│   |	├── backend.rs
		│   |	├── engine.rs
		│   |	├── inflight.rs
		│   |	├── lib.rs
		│   |	└── worker.rs
		│   |
		│   └── Cargo.toml
		│
		├── client-gpu/
		│   ├── src
		│   |	├── cuda/
		|	|	|	├── kernels.ptx
		|	|	|	└── mod.rs
		│   |	|
		│   |	├── opencl/
		|	|	|	└── mod.rs
		│   |	|
		│   |	├── detect.rs
		│   |	├── error.rs
		│   |	└── lib.rs
		│   |
		│   └── Cargo.toml
		│
		└── client-client-gui/
			├── capabilities/
			|	└── default.json
			|
			├── gen/
			|	└── schemas/
			|		├── acl-manifests.json
			|		├── capabilities.json
			|		├── desktop-schema.json
			|		├── linux-schema.json
			|		└── windows-schema.json
			|
			├── icons/
			|	├── icon.ico
			|	└── icon.png
			|
			├── src/
			|	└── main.rs
			|
			├── build.rs
			├── Cargo.toml
			└── tauri.conf.json
