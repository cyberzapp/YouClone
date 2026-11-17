"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Timeline, type RegionRange } from "./timeline";
import { Vault } from "./vault";
import { defaultSettings, describe, type Block, type Format, type Library, type Model, type Project, type Settings } from "./types";

const initialLibrary: Library = { voices: [], samples: [], projects: [], blocks: [], takes: [], sources: [] };
const designPresets = {
  Conversational: "Natural conversational delivery with an easy rhythm and a warm tone.",
  Energetic: "Upbeat, bright delivery with varied intonation and crisp articulation.",
  Documentary: "Calm documentary narration with measured pacing and a grounded tone.",
  Whisper: "Quiet, breathy whispered delivery with clear articulation.",
};
const models: Record<Model, string> = { omni: "OmniVoice", zonos: "ZONOS3-Mini", qwen: "Qwen3 VoiceDesign" };

function makeBlock(projectId: string, text = ""): Block {
  return { id: crypto.randomUUID(), projectId, selected: null, settings: { ...defaultSettings, text } };
}

function synthesize(endpoint: string, settings: object, progress: (message: string) => void,
                    preview: (buffer: ArrayBuffer) => void): Promise<{ id: string; seconds: number }> {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(endpoint);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => socket.send(JSON.stringify(settings));
    socket.onmessage = ({ data }) => {
      if (data instanceof ArrayBuffer) {
        preview(data);
        return;
      }
      const result = JSON.parse(data);
      if (result.type === "progress") progress(`Generating chunk ${result.current} of ${result.total}…`);
      if (result.type === "done") { resolve(result); socket.close(); }
      if (result.type === "error") { reject(new Error(result.message)); socket.close(); }
    };
    socket.onerror = () => reject(new Error("Could not connect to local inference."));
    socket.onclose = () => reject(new Error("Local inference disconnected."));
  });
}

export default function Studio() {
  const [library, setLibrary] = useState(initialLibrary);
  const [project, setProject] = useState<Project>();
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [focused, setFocused] = useState("");
  const [launch, setLaunch] = useState<Awaited<ReturnType<Window["studio"]["service"]>>>();
  const [vault, setVault] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<{ url: string; label: string; master?: { takes: string[]; pause: number } }>();
  const [audition, setAudition] = useState(false);
  const activeTask = useRef(false);
  const player = useRef<(() => void) | null>(null);
  const field = useRef<HTMLTextAreaElement | null>(null);
  const setPlayer = useCallback((next: (() => void) | null) => { player.current = next; }, []);

  useEffect(() => {
    if (!window.studio) return;
    let alive = true;
    void window.studio.library().then(data => {
      if (!alive) return;
      const current = data.projects.at(-1) ?? { id: crypto.randomUUID(), name: "Untitled project" };
      const rows = data.blocks.filter(block => block.projectId === current.id);
      const loaded = rows.length ? rows : [makeBlock(current.id)];
      setLibrary(data);
      setProject(current);
      setBlocks(loaded);
      setFocused(loaded[0].id);
    }).catch(e => { if (alive) setError(describe(e)); });
    const update = () => void window.studio.service().then(value => { if (alive) setLaunch(value); })
      .catch(e => { if (alive) setError(describe(e)); });
    update();
    const interval = setInterval(update, 1200);
    return () => { alive = false; clearInterval(interval); };
  }, []);

  useEffect(() => {
    if (project && !busy) void window.studio.project(project, blocks).catch(e => setError(describe(e)));
  }, [project, blocks, busy]);

  const block = blocks.find(row => row.id === focused);
  const takes = library.takes.filter(take => take.blockId === focused);
  const takeIndex = takes.findIndex(take => take.id === block?.selected);
  const selected = takes[takeIndex];
  const audio = preview?.url ?? (selected ? `/media/takes/${selected.id}.wav` : "");

  function patch(settings: Partial<Settings>, id = focused) {
    setBlocks(rows => rows.map(row => row.id === id ? { ...row, settings: { ...row.settings, ...settings } } : row));
  }
  function focus(id: string) {
    setFocused(id);
    setPreview(undefined);
  }
  function pickTake(id: string, selected: string) {
    setBlocks(rows => rows.map(row => row.id === id ? { ...row, selected } : row));
    setPreview(undefined);
  }
  function cycle(delta: number) {
    if (takes.length > 1) pickTake(focused, takes[(takeIndex + delta + takes.length) % takes.length].id);
  }
  function tag(value: string) {
    if (!block || !field.current) return;
    const { selectionStart } = field.current;
    const text = block.settings.text;
    const addition = `${selectionStart && !/\s/.test(text[selectionStart - 1]) ? " " : ""}${value} `;
    patch({ text: text.slice(0, selectionStart) + addition + text.slice(selectionStart) });
    requestAnimationFrame(() => {
      field.current?.focus();
      field.current?.setSelectionRange(selectionStart + addition.length, selectionStart + addition.length);
    });
  }

  async function task(action: () => Promise<void>) {
    if (activeTask.current) return;
    activeTask.current = true;
    setBusy(true); setError(""); setNotice("");
    try { await action(); }
    catch (e) { setError(describe(e)); setNotice(""); }
    finally { activeTask.current = false; setBusy(false); }
  }

  async function generate(full: boolean) {
    if (!project || !launch?.ready) return;
    const target = full ? blocks.filter(row => row.settings.text.trim()) : block ? [block] : [];
    if (!target.length || !target[0].settings.text.trim()) {
      setError("Enter text in the script editor."); return;
    }
    await task(async () => {
      let context: AudioContext | undefined;
      const sources: AudioBufferSourceNode[] = [];
      let clock = 0;
      if (audition) {
        context = new AudioContext();
        await context.resume();
      }
      function play(buffer: ArrayBuffer) {
        if (!context) return;
        const pcm = new Int16Array(buffer);
        const clip = context.createBuffer(1, pcm.length, 44100);
        clip.copyToChannel(Float32Array.from(pcm, sample => sample / 32768), 0);
        const source = context.createBufferSource();
        source.buffer = clip;
        source.connect(context.destination);
        clock = Math.max(clock, context.currentTime + .04);
        source.start(clock);
        clock += clip.duration;
        sources.push(source);
      }
      try {
        await window.studio.project(project, blocks);
        for (const [index, row] of target.entries()) {
          const sample = library.samples.find(value => value.id === row.settings.sample);
          const settings = {
            ...row.settings, sample: sample?.id ?? null, transcript: sample?.transcript ?? "",
            language: row.settings.language || (row.settings.model !== "qwen" ? sample?.language ?? "" : ""),
            embedding: sample?.embedding ?? null,
          };
          setNotice(`Preparing block ${index + 1} of ${target.length}…`);
          const output = await synthesize(launch.ws, settings, message => setNotice(
            `Block ${index + 1} of ${target.length} · ${message}`,
          ), play);
          const take = { ...output, settings: row.settings, blockId: row.id, created: Date.now() };
          await window.studio.take(take);
          setLibrary(data => ({ ...data, takes: [...data.takes, take] }));
          pickTake(row.id, take.id);
        }
        setNotice(full ? `Generated ${target.length} blocks.` : "Take saved.");
      } finally {
        if (context) {
          const current = context;
          const delay = Math.max(0, (clock - current.currentTime) * 1000);
          window.setTimeout(() => { sources.forEach(source => source.disconnect()); void current.close(); }, delay + 30);
        }
      }
    });
  }

  async function exportAudio(format: Format, subtitles: boolean, master: boolean,
                             region?: RegionRange, quick = false, listen = false) {
    const ids = master ? blocks.filter(row => row.settings.text.trim()).map(row => row.selected) : [selected?.id];
    if (!ids.length || ids.some(id => !id)) {
      setError(master ? "Generate a take for each nonempty block before exporting the master." : "Select a take to export.");
      return;
    }
    await task(async () => {
      setNotice(listen ? "Building master preview…" : "Rendering export…");
      const takes = ids as string[];
      const pause = block?.settings.period_pause ?? .45;
      const result = await window.studio.export({
        takes, format, subtitles, ...(master ? {} : region),
        polish: master && !listen && blocks.some(row => row.settings.polish), pause,
      }, quick, listen);
      if (result.url) setPreview({ url: result.url, label: "Master preview", master: { takes, pause } });
      setNotice(result.destination ? `Exported to ${result.destination}` : "");
    });
  }

  useEffect(() => {
    function handle(event: KeyboardEvent) {
      if (event.defaultPrevented || event.repeat) return;
      const typing = (event.target as HTMLElement | null)?.closest("textarea,input,select,[contenteditable],button");
      if (event.code === "Space" && !typing && !vault && player.current) {
        event.preventDefault(); player.current(); return;
      }
      if (!event.metaKey) return;
      const key = event.key.toLowerCase();
      if (!["enter", "r", "[", "]", "e"].includes(key)) return;
      event.preventDefault();
      if (busy || vault) return;
      if (key === "enter") void generate(event.shiftKey);
      if (key === "r") setVault(true);
      if (key === "[") cycle(-1);
      if (key === "]") cycle(1);
      if (key === "e") void exportAudio("wav", false, true, undefined, true);
    }
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  });

  const model = block?.settings.model;
  return <main className="studio">
    <header className="topbar">
      <div className="brand"><span className="brand-mark" aria-hidden="true">Y</span><div><strong>YouClone</strong><span>Voice production</span></div></div>
      <span className={launch?.ready ? "status online" : "status"}><i />{launch?.ready ? "Local inference" : launch?.serviceError ? "Sidecar unavailable" : "Starting sidecar"}</span>
      <button className="accent" disabled={!launch?.ready || busy} onClick={() => void generate(true)}>Generate script <kbd>⌘⇧↵</kbd></button>
    </header>
    <div className="workbench">
      <aside className="sidebar">
        <div className="section-heading"><h2>Voice vault</h2><button disabled={busy} onClick={() => setVault(true)}>Add sample</button></div>
        {library.voices.map(voice => <section className="voice-group" key={voice.id}>
          <h3>{voice.name}</h3>
          {library.samples.filter(sample => sample.voiceId === voice.id).map(sample =>
            <button key={sample.id} className="source" disabled={busy || !block || model === "qwen"}
              aria-pressed={block?.settings.sample === sample.id} onClick={() => {
                patch({ sample: sample.id });
                setPreview({ url: `/media/samples/${sample.id}.wav`, label: `${voice.name} · ${sample.style}` });
              }}><strong>{sample.style}</strong><small>{sample.language.toUpperCase()} · {sample.seconds.toFixed(1)} s</small></button>)}
        </section>)}
        {!!library.sources.length && <section className="voice-group"><h3>Workspace recordings</h3>
          {library.sources.map(source => <button key={source} className="source"
            onClick={() => setPreview({ url: `/media/sources/${encodeURIComponent(source)}`, label: source })}>{source}</button>)}
        </section>}
        {!library.voices.length && <p className="hint">Add a reference clip to start cloning a voice.</p>}
      </aside>
      <section className="editor" aria-label="Script editor">
        <div className="section-heading">
          <input className="project-title" aria-label="Project name" value={project?.name ?? ""} disabled={busy || !project}
            onChange={event => project && setProject({ ...project, name: event.target.value })} />
          <button disabled={!project || busy} onClick={() => {
            if (!project) return;
            const row = makeBlock(project.id);
            setBlocks([...blocks, row]); focus(row.id);
          }}>Add block</button>
        </div>
        {blocks.map((row, index) => <article className="script-block" data-active={row.id === focused} key={row.id}>
          <div className="block-heading"><button className="block-name" onClick={() => focus(row.id)}>Block {index + 1}</button>
            <span>{models[row.settings.model]}</span>
            <button disabled={busy || blocks.length < 2} aria-label={`Remove block ${index + 1}`} onClick={() => {
              const next = blocks.filter(value => value.id !== row.id);
              setBlocks(next);
              if (focused === row.id) focus(next[Math.max(0, index - 1)].id);
            }}>Remove</button>
          </div>
          <textarea ref={row.id === focused ? field : undefined} disabled={busy} aria-label={`Block ${index + 1} script`}
            placeholder="Type or paste your script…" value={row.settings.text} onFocus={() => { if (row.id !== focused) focus(row.id); }}
            onChange={event => patch({ text: event.target.value }, row.id)}
            onPaste={event => {
              const parts = event.clipboardData.getData("text").split(/\r?\n\s*\r?\n/).map(text => text.trim()).filter(Boolean);
              if (!project || row.settings.text || parts.length < 2) return;
              event.preventDefault();
              const inserted = parts.slice(1).map(text => makeBlock(project.id, text));
              setBlocks([...blocks.slice(0, index), { ...row, settings: { ...row.settings, text: parts[0] } }, ...inserted, ...blocks.slice(index + 1)]);
            }} />
          {row.id === focused && <div className="block-tools">
            {model === "omni" && <><button disabled={busy} onClick={() => tag("[sigh]")}>[sigh]</button>
              <button disabled={busy} onClick={() => tag("[laughter]")}>[laughter]</button>
              <button disabled={busy} onClick={() => tag("[surprise-ah]")}>[surprise-ah]</button>
              <button disabled={busy} onClick={() => tag("[question-oh]")}>[question-oh]</button></>}
            <button disabled={busy} onClick={() => tag("[pause 0.5s]")}>[pause 0.5s]</button>
            <button className="accent regenerate" disabled={busy || !launch?.ready || !row.settings.text.trim()}
              onClick={() => void generate(false)}>{row.selected ? "Regenerate block" : "Generate block"} <kbd>⌘↵</kbd></button>
          </div>}
        </article>)}
        {(error || launch?.serviceError || launch?.modelWarning) && <p role="alert" className="error">{error || launch?.serviceError || launch?.modelWarning}</p>}
        <p aria-live="polite" className="notice">{notice}</p>
      </section>
      <aside className="inspector">
        <div className="section-heading"><h2>Block settings</h2></div>
        {block && <fieldset disabled={busy}>
          <label>Model<select value={model} onChange={event => patch({ model: event.target.value as Model, language: "", style: "" })}>
            {Object.entries(models).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
          </select></label>
          {model !== "qwen" && <label>Reference sample<select value={block.settings.sample} onChange={event => patch({ sample: event.target.value })}>
            <option value="">Select a sample</option>
            {library.voices.map(voice => <optgroup key={voice.id} label={voice.name}>
              {library.samples.filter(sample => sample.voiceId === voice.id).map(sample =>
                <option key={sample.id} value={sample.id}>{sample.style} · {sample.seconds.toFixed(1)} s</option>)}
            </optgroup>)}
          </select></label>}
          {model === "omni" && <p className="hint">
            Choose a clip of your voice with the delivery you want to hear.
          </p>}
          <label>Language<input value={block.settings.language} placeholder={model === "qwen" ? "Auto" : "Reference language"}
              onChange={event => patch({ language: event.target.value })} /></label>
          {model === "qwen" && <label>Delivery preset<select value="" onChange={event => {
            const preset = event.target.value as keyof typeof designPresets;
            if (preset) patch({ style: designPresets[preset] });
          }}>
            <option value="">Choose a preset</option>
            {Object.keys(designPresets).map(preset => <option key={preset}>{preset}</option>)}
          </select></label>}
          {model !== "zonos" && <label>{model === "qwen" ? "Voice design prompt" : "Style instruction"}<textarea rows={2} value={block.settings.style}
              onChange={event => patch({ style: event.target.value })} placeholder={model === "qwen" ? "Describe tone, pitch, and delivery" : "Optional model style"} /></label>}
          <label>Speed <output>{block.settings.speed.toFixed(2)}×</output><input type="range" min=".7" max="1.3" step=".01"
            value={block.settings.speed} onChange={event => patch({ speed: event.target.valueAsNumber })} /></label>
          <label>Temperature <output>{block.settings.temperature.toFixed(2)}</output><input type="range" min="0" max="2" step=".05"
            value={block.settings.temperature} onChange={event => patch({ temperature: event.target.valueAsNumber })} /></label>
          <details><summary>Advanced</summary>
            {model !== "qwen" && <label>Guidance<input type="number" min="0" max="8" step=".1" value={block.settings.guidance} onChange={event => patch({ guidance: event.target.valueAsNumber || 0 })} /></label>}
            {model !== "omni" && <label>Top p<input type="number" min=".01" max="1" step=".01" value={block.settings.top_p} onChange={event => patch({ top_p: event.target.valueAsNumber || .95 })} /></label>}
            {model !== "omni" && <label>Repetition penalty<input type="number" min="1" max="2" step=".01" value={block.settings.repetition_penalty} onChange={event => patch({ repetition_penalty: event.target.valueAsNumber || 1 })} /></label>}
            <label>Seed<input type="number" min="0" max="4294967295" value={block.settings.seed} onChange={event => patch({ seed: event.target.valueAsNumber || 0 })} /></label>
            <label>Comma pause (s)<input type="number" min="0" max="3" step=".05" value={block.settings.comma_pause} onChange={event => patch({ comma_pause: event.target.valueAsNumber || 0 })} /></label>
            <label>Sentence pause (s)<input type="number" min="0" max="3" step=".05" value={block.settings.period_pause} onChange={event => patch({ period_pause: event.target.valueAsNumber || 0 })} /></label>
          </details>
          <label className="check"><input type="checkbox" checked={block.settings.polish} onChange={event => patch({ polish: event.target.checked })} /> Studio Polish</label>
          <label className="check"><input type="checkbox" checked={audition} onChange={event => setAudition(event.target.checked)} /> Audition while generating</label>
        </fieldset>}
        <div className="takes">
          <button disabled={busy || takes.length < 2} aria-label="Previous take" onClick={() => cycle(-1)}>‹</button>
          <span>{selected ? `Take ${takeIndex + 1} / ${takes.length}` : "No takes"}</span>
          <button disabled={busy || takes.length < 2} aria-label="Next take" onClick={() => cycle(1)}>›</button>
        </div>
        <button disabled={busy || !selected} onClick={() => setPreview(undefined)}>Show selected take</button>
        <button className="master" disabled={busy || !launch?.ready || !blocks.some(row => row.selected)}
          onClick={() => void exportAudio("wav", false, true, undefined, false, true)}>Preview master</button>
      </aside>
    </div>
    <Timeline key={audio} audio={audio} label={preview?.label ?? (selected ? `Block ${blocks.findIndex(row => row.id === focused) + 1} · Take ${takeIndex + 1}` : "")}
      locked={busy || !launch?.ready} canExportTake={preview ? !!preview.master : !!selected}
      canExportMaster={blocks.some(row => !!row.settings.text.trim()) &&
        blocks.every(row => !row.settings.text.trim() || !!row.selected)}
      setPlayer={setPlayer} exportAudio={(format, subtitles, master, region) => {
          if (!master && preview?.master) {
          const { takes, pause } = preview.master;
          void task(async () => {
            const result = await window.studio.export({ takes, format, subtitles, ...region, pause }, false, false);
            if (result.destination) setNotice(`Exported to ${result.destination}`);
          });
          return;
        }
        void exportAudio(format, subtitles, master, region);
      }} />
    {vault && <Vault sources={library.sources} voices={library.voices} ready={!!launch?.ready} close={() => setVault(false)}
      saved={(voice, sample) => {
        setLibrary(data => ({ ...data, voices: data.voices.some(item => item.id === voice.id) ? data.voices : [...data.voices, voice],
          samples: [...data.samples, sample] }));
        if (model !== "qwen") patch({ sample: sample.id });
        setPreview({ url: `/media/samples/${sample.id}.wav`, label: `${voice.name} · ${sample.style}` });
        setVault(false);
      }} />}
  </main>;
}
