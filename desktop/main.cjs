const { app, BrowserWindow, session, protocol, net, ipcMain, dialog, nativeImage } = require("electron");
const { randomBytes } = require("node:crypto");
const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const { createServer } = require("node:net");
const { copyFile, mkdir, readdir, stat } = require("node:fs/promises");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const openStore = require("./store.cjs");

app.setName("YouClone");
const previousUserData = path.join(app.getPath("appData"), "VoxM5 Studio");
if (!existsSync(app.getPath("userData")) && existsSync(previousUserData)) {
  app.setPath("userData", previousUserData);
}
const root = process.env.YOUCLONE_ROOT
  ? path.resolve(process.env.YOUCLONE_ROOT)
  : path.resolve(__dirname, "..");
const origin = "voxm5://studio";
const uuid = "[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}";
const outputName = new RegExp(`^${uuid}\\.(wav|mp3|flac|m4a|srt)$`, "i");
const voicePreviewName = /^[a-z0-9_-]{3,64}\.wav$/i;
const isWin = process.platform === "win32";
const isMac = process.platform === "darwin";
const voiceCorpusDir = path.join(root, "preserved-for-future", "voice-corpus");
const safeAudioName = /^[a-z0-9 _.-]{1,160}\.(wav|mp3|m4a|flac|ogg)$/i;

function isAllowedSourceName(name) {
  return typeof name === "string" && safeAudioName.test(name) && !name.includes("..") && !name.includes("/") && !name.includes("\\");
}

function resolveSourcePath(name) {
  const archived = path.join(voiceCorpusDir, name);
  return existsSync(archived) ? archived : path.join(root, name);
}

async function listAvailableSources() {
  const found = new Set();
  for (const dir of [voiceCorpusDir, root]) {
    if (!existsSync(dir)) continue;
    try {
      for (const name of await readdir(dir)) {
        if (isAllowedSourceName(name)) found.add(name);
      }
    } catch {}
  }
  return [...found];
}
const token = randomBytes(32).toString("hex");
let sidecar, port, store, data, win;
let ready = false;

protocol.registerSchemesAsPrivileged([{
  scheme: "voxm5",
  privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true },
}]);

function freePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(error => error ? reject(error) : resolve(port));
    });
  });
}

async function backend(route, options = {}) {
  if (!ready && route !== "health") throw new Error("The inference service is still starting.");
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return fetch(`http://127.0.0.1:${port}/${route}`, { ...options, headers });
}

const UPLOAD_ROUTES = {
  "/api/sample": "sample",
  "/api/transcribe": "transcribe",
  "/api/convert": "convert",
  "/api/isolate": "isolate",
  "/api/watermark/detect": "watermark/detect",
  "/api/studio-voices/instant": "studio-voices/instant",
  "/api/studio-voices/professional": "studio-voices/professional",
};

const JSON_ROUTES = {
  "/api/sfx": "sfx",
  "/api/tools/direction": "tools/direction",
  "/api/tools/rate-fit": "tools/rate-fit",
  "/api/models/download": "models/download",
  "/api/lora/train": "lora/train",
  "/api/studio-voices/design/preview": "studio-voices/design/preview",
  "/api/studio-voices/design/save": "studio-voices/design/save",
};

const GET_ROUTES = {
  "/api/models/status": "models/status",
  "/api/lora/status": "lora/status",
  "/api/studio-voices": "studio-voices",
};

async function serve(request) {
  const url = new URL(request.url);
  if (url.hostname !== "studio") return new Response(null, { status: 404 });

  // DELETE /api/studio-voices/<id>
  const delMatch = url.pathname.match(/^\/api\/studio-voices\/([a-z0-9_-]{3,64})$/i);
  if (delMatch && request.method === "DELETE") {
    try {
      return await backend(`studio-voices/${delMatch[1]}`, { method: "DELETE" });
    } catch (error) {
      return Response.json({ detail: error.message }, { status: 503 });
    }
  }

  const getRoute = GET_ROUTES[url.pathname] ?? null;
  if (getRoute && request.method === "GET") {
    try {
      return await backend(getRoute, { method: "GET" });
    } catch (error) {
      return Response.json({ detail: error.message }, { status: 503 });
    }
  }

  const uploadRoute = UPLOAD_ROUTES[url.pathname] ?? null;
  if (uploadRoute && request.method === "POST") {
    if (Number(request.headers.get("Content-Length")) > 1060 * 1024 * 1024) {
      return Response.json({ detail: "Upload exceeds 1 GB limit." }, { status: 413 });
    }
    try {
      const body = await request.formData();
      const name = body.get("source");
      if (name) {
        if (!isAllowedSourceName(name)) return new Response(null, { status: 404 });
        const resolved = resolveSourcePath(name);
        if (!existsSync(resolved)) return new Response(null, { status: 404 });
        const { openAsBlob } = require("node:fs");
        body.set("file", await openAsBlob(resolved), name);
        body.delete("source");
      }
      return await backend(uploadRoute, { method: "POST", body });
    } catch (error) {
      return Response.json({ detail: error.message }, { status: 503 });
    }
  }

  const jsonRoute = JSON_ROUTES[url.pathname] ?? null;
  if (jsonRoute && request.method === "POST") {
    try {
      const text = await request.text();
      return await backend(jsonRoute, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: text,
      });
    } catch (error) {
      return Response.json({ detail: error.message }, { status: 503 });
    }
  }

  const pathname = decodeURIComponent(url.pathname);
  const voicePrevMatch = pathname.match(/^\/media\/voices\/([^/]+)$/);
  const match = pathname.match(/^\/media\/(sources|samples|takes|exports)\/([^/]+)$/);
  let file;
  if (voicePrevMatch) {
    const fname = voicePrevMatch[1];
    if (!voicePreviewName.test(fname)) return new Response(null, { status: 404 });
    const vid = fname.replace(/\.wav$/i, "");
    file = path.join(data, "voices_studio", vid, "preview.wav");
  } else if (match) {
    const [, bucket, name] = match;
    const valid = bucket === "sources" ? isAllowedSourceName(name) : outputName.test(name);
    if (!valid) return new Response(null, { status: 404 });
    file = bucket === "sources" ? resolveSourcePath(name) : path.join(data, bucket, name);
  } else {
    const out = path.join(root, "out");
    file = path.resolve(out, "." + (pathname === "/" ? "/index.html" : pathname));
    if (!file.startsWith(out + path.sep)) return new Response(null, { status: 404 });
  }
  try {
    if (!(await stat(file)).isFile()) return new Response(null, { status: 404 });
    const range = request.headers.get("range");
    const response = await net.fetch(pathToFileURL(file).href, range ? { headers: { range } } : {});
    const headers = new Headers(response.headers);
    headers.set("Content-Security-Policy",
      `default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self' ws://127.0.0.1:${port}; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`);
    return new Response(response.body, { status: response.status, headers });
  } catch {
    return new Response(null, { status: 404 });
  }
}

function register(channel, listener) {
  ipcMain.handle(channel, (event, ...args) => {
    if (event.sender !== win?.webContents || !event.senderFrame?.url.startsWith(origin + "/")) {
      throw new Error("Unexpected caller.");
    }
    return listener(...args);
  });
}

async function start() {
  if (!app.requestSingleInstanceLock()) return app.quit();
  const iconPath = path.join(root, "desktop", "assets", "icon.png");
  if (process.platform === "darwin" && existsSync(iconPath) && app.dock) {
    try { app.dock.setIcon(nativeImage.createFromPath(iconPath)); } catch {}
  }

  data = path.join(app.getPath("userData"), "studio");
  await mkdir(data, { recursive: true });
  store = openStore(path.join(data, "studio.sqlite"));
  port = await freePort();
  const python = isWin
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");
  const toolsBin = path.join(root, ".voxm5", "tools");
  const extraPaths = isWin
    ? [toolsBin]
    : [toolsBin, "/opt/homebrew/bin", "/usr/local/bin"];
  const envPath = [...extraPaths, process.env.PATH || ""].filter(Boolean).join(path.delimiter);
  const hfHome = path.join(root, ".voxm5", "huggingface");
  const hasCachedModels = existsSync(path.join(hfHome, "hub"));
  sidecar = spawn(python, ["-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1",
    "--port", String(port), "--no-access-log"], {
    cwd: root,
    stdio: "inherit",
    env: {
      ...process.env,
      PATH: envPath,
      VOXM5_DATA: data,
      VOXM5_TOKEN: token,
      HF_HOME: hfHome,
      ...(hasCachedModels ? { HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" } : {}),
    },
  });
  let serviceError, modelWarning;
  sidecar.on("error", error => { serviceError = error.message; });
  sidecar.on("exit", code => { ready = false; serviceError = `Inference service exited (${code}).`; });
  void (async () => {
    for (let attempt = 0; attempt < 300 && !serviceError; attempt++) {
      try {
        const response = await backend("health");
        if (response.ok) {
          ready = true;
          modelWarning = (await response.json()).modelWarning;
          return;
        }
      } catch {}
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    serviceError ??= "Inference service did not start.";
  })();

  protocol.handle("voxm5", serve);
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    callback(contents === win?.webContents && permission === "media" &&
      details.requestingUrl.startsWith(origin + "/") &&
      details.mediaTypes?.includes("audio") && !details.mediaTypes?.includes("video"));
  });
  session.defaultSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) =>
    contents === win?.webContents && permission === "media" &&
    requestingOrigin.replace(/\/$/, "") === origin && details.mediaType !== "video");

  register("library", async () => ({ ...store.read(), sources: await listAvailableSources() }));
  register("service", async () => {
    if (ready) {
      try { modelWarning = (await (await backend("health")).json()).modelWarning; }
      catch {}
    }
    return { ready, serviceError, modelWarning, ws: `ws://127.0.0.1:${port}/generate?token=${token}` };
  });
  register("project", (project, blocks) => store.saveProject(project, blocks));
  register("sample", (name, sample) => store.saveSample(name, sample));
  register("take", take => store.saveTake(take));
  register("export", async (settings, quick, preview) => {
    const response = await backend("export", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Invalid export settings.");
    if (preview) return { url: `/media/exports/${result.file}` };
    let destination;
    if (quick) {
      const preferred = process.env.VOXM5_WATCH_FOLDER || path.join(app.getPath("music"), "YouClone Exports");
      await mkdir(preferred, { recursive: true });
      destination = path.join(preferred, `YouClone-${result.file}`);
    } else {
      const selection = await dialog.showSaveDialog(win, {
        defaultPath: `YouClone.${settings.format}`, filters: [{ name: settings.format.toUpperCase(), extensions: [settings.format] }],
      });
      if (selection.canceled) return {};
      destination = selection.filePath;
    }
    await copyFile(path.join(data, "exports", result.file), destination);
    if (result.subtitles) {
      const parsed = path.parse(destination);
      await copyFile(path.join(data, "exports", result.subtitles), path.join(parsed.dir, parsed.name + ".srt"));
    }
    return { destination };
  });

  win = new BrowserWindow({
    title: "YouClone", width: 1440, height: 950, minWidth: 880, minHeight: 650,
    icon: existsSync(iconPath) ? iconPath : undefined,
    ...(isMac ? { titleBarStyle: "hiddenInset", trafficLightPosition: { x: 16, y: 18 } } : {}),
    backgroundColor: "#090A0F", webPreferences: { preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(origin + "/")) event.preventDefault();
  });
  await win.loadURL(origin + "/");
}

app.whenReady().then(start).catch(error => {
  dialog.showErrorBox("YouClone", error.message);
  app.quit();
});
app.on("second-instance", () => {
  if (win?.isMinimized()) win.restore();
  win?.focus();
});
app.on("window-all-closed", () => app.quit());
app.on("before-quit", () => {
  sidecar?.kill();
  store?.close();
});
