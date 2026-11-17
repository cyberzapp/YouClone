const { app, BrowserWindow, session, protocol, net, ipcMain, dialog } = require("electron");
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
const root = path.resolve(__dirname, "..");
const origin = "voxm5://studio";
const uuid = "[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}";
const outputName = new RegExp(`^${uuid}\\.(wav|mp3|flac|srt)$`, "i");
const sources = ["English_voice2.wav", "Hindi_voice.wav", "english.mp3"];
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

async function serve(request) {
  const url = new URL(request.url);
  if (url.hostname !== "studio") return new Response(null, { status: 404 });
  if (url.pathname === "/api/sample" && request.method === "POST") {
    if (Number(request.headers.get("Content-Length")) > 158 * 1024 * 1024) {
      return Response.json({ detail: "Recording exceeds 150 MB." }, { status: 413 });
    }
    try {
      const body = await request.formData();
      const name = body.get("source");
      if (name) {
        if (!sources.includes(name)) return new Response(null, { status: 404 });
        const { openAsBlob } = require("node:fs");
        body.set("file", await openAsBlob(path.join(root, name)), name);
        body.delete("source");
      }
      return await backend("sample", { method: "POST", body });
    } catch (error) {
      return Response.json({ detail: error.message }, { status: 503 });
    }
  }
  const pathname = decodeURIComponent(url.pathname);
  const match = pathname.match(/^\/media\/(sources|samples|takes|exports)\/([^/]+)$/);
  let file;
  if (match) {
    const [, bucket, name] = match;
    const valid = bucket === "sources" ? sources.includes(name) : outputName.test(name);
    if (!valid) return new Response(null, { status: 404 });
    file = path.join(bucket === "sources" ? root : path.join(data, bucket), name);
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
  data = path.join(app.getPath("userData"), "studio");
  await mkdir(data, { recursive: true });
  store = openStore(path.join(data, "studio.sqlite"));
  port = await freePort();
  const python = path.join(root, ".venv", "bin", "python");
  sidecar = spawn(python, ["-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1",
    "--port", String(port), "--no-access-log"], {
    cwd: root, stdio: "inherit", env: { ...process.env, VOXM5_DATA: data, VOXM5_TOKEN: token,
      HF_HOME: path.join(root, ".voxm5", "huggingface"), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" },
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

  register("library", async () => ({ ...store.read(), sources: (await readdir(root)).filter(name => sources.includes(name)) }));
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
    title: "YouClone", width: 1440, height: 950, minWidth: 860, minHeight: 650,
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
