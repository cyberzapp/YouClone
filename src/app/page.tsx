"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  IconBolt,
  IconChevronDown,
  IconChevronLeft,
  IconChevronRight,
  IconClose,
  IconDocument,
  IconFolderPlus,
  IconHistory,
  IconLab,
  IconMic,
  IconPlus,
  IconSliders,
  IconSoundwave,
  IconWand,
  IconWaveform,
} from "./icons";
import { Timeline, type RegionRange } from "./timeline";
import { Vault } from "./vault";
import { VoiceCreationModal } from "./voice_creation";
import {
  SoundEffectsStudio,
  StudioToolsModal,
  TakeHistory,
  Transcription,
  VoiceChanger,
  type Caption,
} from "./production";
import { applyPronunciation } from "./pronunciation";
import {
  defaultSettings,
  describe,
  type AmbienceMode,
  type Block,
  type DistanceMode,
  type EffectPreset,
  type ExportSettings,
  type FilterMode,
  type Format,
  type Library,
  type LoraAdapter,
  type Model,
  type ModelStatusInfo,
  type PauseFill,
  type Project,
  type PvcTrainingState,
  type Settings,
  type SpaceMode,
  type StudioVoice,
} from "./types";

const initialLibrary: Library = { voices: [], samples: [], projects: [], blocks: [], takes: [], sources: [] };
const designPresets = {
  Conversational: "Natural conversational delivery with an easy rhythm and a warm tone.",
  Energetic: "Upbeat, bright delivery with varied intonation and crisp articulation.",
  Documentary: "Calm documentary narration with measured pacing and a grounded tone.",
  Whisper: "Quiet, breathy whispered delivery with clear articulation.",
  "Cinematic Trailer": "Deep, resonant cinematic trailer narration with dramatic gravitas and crisp diction.",
  "Audiobook Narrator": "Expressive literary audiobook narrator with warm midrange and natural storytelling cadence.",
};
const starterPrompts: Record<string, { text: string; style: string }> = {
  "Viral Short Hook": {
    text: "Wait... people are already playing GTA 6 on PC?!\n\nBro, the game's not even out on PC yet!\n\nSo how are they doing this?\n\nWell, turns out there's actually a way to play GTA 6 on your computer right now. But... there's one tiny problem.\n\nYeah, <sigh> sorry bro. Not happening. <chuckle> Unless you already own a console!",
    style: "",
  },
  "Dramatic Story": {
    text: "Listen carefully... every decision we make in the next five minutes determines how this story ends.",
    style: "",
  },
  "Expressive Sigh": {
    text: "You knew the whole time, didn't you? <sigh> And you still let me walk right into the trap.",
    style: "",
  },
  "Comedic Punchline": {
    text: "Imagine waiting all these years for GTA 6... <chuckle> only to end up playing GTA 6: Password Theft Edition!",
    style: "",
  },
};
const models: Record<Model, string> = {
  orpheus_seedvc: "YouClone Studio (Orpheus-3B + Seed-VC 44kHz)",
};

type DefaultVoiceManifest = {
  voice?: { id: string; name: string };
  lora_adapter?: string;
  english_sample_id?: string;
  english_alt_sample_id?: string;
  hindi_sample_id?: string;
};

function migrateBlockSettings(saved: Partial<Settings>): Settings {
  const merged = { ...defaultSettings, ...saved };
  if (saved.model && saved.model !== "orpheus_seedvc") {
    merged.model = "orpheus_seedvc";
    merged.temperature = defaultSettings.temperature;
    merged.top_p = defaultSettings.top_p;
    merged.repetition_penalty = defaultSettings.repetition_penalty;
    if (saved.seed === 42) merged.seed = defaultSettings.seed;
  }
  return merged;
}

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
      if (result.type === "progress") {
        if (result.emotion && String(result.emotion).startsWith("Stage ")) {
          progress(`${result.emotion}…`);
        } else {
          const emo = result.emotion && result.emotion !== "warm_explain" && result.emotion !== "neutral"
            ? ` · ${String(result.emotion).replace(/_/g, " ")}`
            : "";
          progress(`Beat ${result.current}/${result.total}${emo}…`);
        }
      }
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
  const [panel, setPanel] = useState<"transcription" | "history" | "changer" | "sfx" | "tools" | "lora">();
  const [showSettings, setShowSettings] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [adapters, setAdapters] = useState<LoraAdapter[]>([]);
  const [modelsStatus, setModelsStatus] = useState<ModelStatusInfo[]>([]);
  const [studioVoices, setStudioVoices] = useState<StudioVoice[]>([]);
  const [pvcTraining, setPvcTraining] = useState<PvcTrainingState>();
  const [voiceModal, setVoiceModal] = useState(false);
  const [defaultVoice, setDefaultVoice] = useState<DefaultVoiceManifest>();
  const [vault, setVault] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progressLabel, setProgressLabel] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [autoPlayNext, setAutoPlayNext] = useState(false);
  const [preview, setPreview] = useState<{ url: string; label: string; master?: Pick<ExportSettings, "takes" | "pause" | "timing" | "effect_preset" | "pause_fill"> }>();
  const [audition, setAudition] = useState(false);
  const activeTask = useRef(false);
  const player = useRef<(() => void) | null>(null);
  const field = useRef<HTMLTextAreaElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const setPlayer = useCallback((next: (() => void) | null) => { player.current = next; }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const onPointer = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    window.addEventListener("mousedown", onPointer);
    return () => window.removeEventListener("mousedown", onPointer);
  }, [menuOpen]);

  useEffect(() => {
    if (!window.studio) return;
    let alive = true;
    void window.studio.library().then(data => {
      if (!alive) return;
      const current = data.projects.at(-1) ?? { id: crypto.randomUUID(), name: "Voiceover Studio" };
      const rows = data.blocks.filter(block => block.projectId === current.id);
      const loaded = rows.length ? rows.map(r => ({ ...r, settings: migrateBlockSettings(r.settings) })) : [makeBlock(current.id)];
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

  const refreshModelsStatus = useCallback(async () => {
    try {
      const [mRes, vRes] = await Promise.all([
        fetch("/api/models/status"),
        fetch("/api/studio-voices"),
      ]);
      if (mRes.ok) {
        const data = await mRes.json();
        if (Array.isArray(data.models)) setModelsStatus(data.models);
        if (Array.isArray(data.adapters)) setAdapters(data.adapters);
        if (data.default_voice) setDefaultVoice(data.default_voice);
        if (Array.isArray(data.studio_voices)) setStudioVoices(data.studio_voices);
        if (data.pvc_training) setPvcTraining(data.pvc_training);
      }
      if (vRes.ok) {
        const vData = await vRes.json();
        if (Array.isArray(vData.voices)) {
          setStudioVoices(vData.voices);
          const defVoice = vData.voices[0]?.id || "";
          setBlocks(rows => rows.map(row =>
            row.settings.studio_voice_id && vData.voices.some((v: StudioVoice) => v.id === row.settings.studio_voice_id)
              ? row
              : { ...row, settings: { ...row.settings, studio_voice_id: defVoice } }
          ));
        }
        if (vData.training) setPvcTraining(vData.training);
      }
    } catch {}
  }, []);

  useEffect(() => {
    if (launch?.ready) void refreshModelsStatus();
  }, [launch?.ready, refreshModelsStatus]);

  useEffect(() => {
    if (project && !busy) void window.studio.project(project, blocks).catch(e => setError(describe(e)));
  }, [project, blocks, busy]);

  const block = blocks.find(row => row.id === focused) ?? blocks[0];
  const takes = library.takes.filter(take => take.blockId === block?.id);
  const takeIndex = takes.findIndex(take => take.id === block?.selected);
  const selected = takes[takeIndex];
  const audio = preview?.url ?? (selected ? `/media/takes/${selected.id}.wav` : "");

  const scriptMetrics = useMemo(() => {
    const fullText = (block?.settings.text || "").trim();
    const clean = fullText.replace(/\[[^\]]*\]|<[^>]*>/g, "").trim();
    const words = clean ? clean.split(/\s+/).length : 0;
    const chars = clean.length;
    const estSeconds = Math.round(words / 2.6);
    const mins = Math.floor(estSeconds / 60);
    const secs = estSeconds % 60;
    return { words, chars, clock: `${mins}:${String(secs).padStart(2, "0")}` };
  }, [block?.settings.text]);

  async function openProject(id?: string, captions?: Caption[]) {
    if (!project) return;
    await task(async () => {
      await window.studio.project(project, blocks);
      const fresh = await window.studio.library();
      const next = id ? fresh.projects.find(item => item.id === id) :
        { id: crypto.randomUUID(), name: captions ? "Timed narration" : "Untitled project" };
      if (!next) throw new Error("Project could not be opened.");
      const saved = fresh.blocks.filter(item => item.projectId === next.id).map(r => ({
        ...r,
        settings: migrateBlockSettings(r.settings),
      }));
      const nextBlocks = captions?.map(({ text, start, end }) => {
        const row = makeBlock(next.id, text);
        if (block) row.settings = { ...defaultSettings, ...block.settings, text };
        row.settings.timing = { start, end };
        return row;
      }) ?? (saved.length ? saved : [{ ...makeBlock(next.id), settings: { ...defaultSettings, sample: block?.settings.sample ?? "" } }]);
      await window.studio.project(next, nextBlocks);
      setLibrary({ ...fresh, projects: id ? fresh.projects : [...fresh.projects, next] });
      setProject(next);
      setBlocks(nextBlocks);
      focus(nextBlocks[0].id);
    });
  }

  function appendScript(text: string) {
    if (!project || !block) return;
    const combined = block.settings.text.trim() ? `${block.settings.text.trim()}\n\n${text.trim()}` : text.trim();
    patch({ text: combined });
  }

  function patch(settings: Partial<Settings>, id = block?.id ?? focused) {
    setBlocks(rows => rows.map(row => row.id === id ? { ...row, settings: { ...defaultSettings, ...row.settings, ...settings } } : row));
  }
  function focus(id: string) {
    setFocused(id);
    setAutoPlayNext(false);
    setPreview(undefined);
  }
  function pickTake(id: string, selectedTakeId: string, autoPlay = false) {
    setAutoPlayNext(autoPlay);
    setBlocks(rows => rows.map(row => row.id === id ? { ...row, selected: selectedTakeId } : row));
    setPreview(undefined);
  }
  function cycle(delta: number) {
    if (takes.length > 1 && block) pickTake(block.id, takes[(takeIndex + delta + takes.length) % takes.length].id, false);
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

  async function attachExternalTake(takeId: string, seconds: number, text: string, settingsPatch: Partial<Settings>) {
    if (!project || !block) return;
    const updatedSettings: Settings = {
      ...defaultSettings,
      ...block.settings,
      ...settingsPatch,
      text: block.settings.text.trim() ? block.settings.text : text,
    };
    const newTake = {
      id: takeId,
      blockId: block.id,
      seconds,
      created: Date.now(),
      settings: updatedSettings,
    };
    await window.studio.project(project, blocks);
    await window.studio.take(newTake);
    setAutoPlayNext(true);
    setBlocks(rows => rows.map(r => r.id === block.id ? { ...r, selected: takeId, settings: updatedSettings } : r));
    setLibrary(data => ({ ...data, takes: [...data.takes, newTake] }));
    setPreview(undefined);
    setNotice("Audio ready.");
  }

  async function task(action: () => Promise<void>) {
    if (activeTask.current) return;
    activeTask.current = true;
    setBusy(true); setError(""); setNotice(""); setProgressLabel("");
    try { await action(); }
    catch (e) { setError(describe(e)); setNotice(""); setProgressLabel(""); }
    finally { activeTask.current = false; setBusy(false); setProgressLabel(""); }
  }

  async function generate(full: boolean, missingOnly = false) {
    if (!project || !launch?.ready) return;
    const target = full ? blocks.filter(row => row.settings.text.trim() && (!missingOnly || !row.selected)) : block ? [block] : [];
    if (!target.length || !target[0].settings.text.trim()) {
      setError(missingOnly ? "Every block already has audio." : "Paste or type your script first."); return;
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
          let rawText = row.settings.text;
          let inlineVoiceSampleId = row.settings.sample || defaultVoice?.english_sample_id || library.samples[0]?.id || "";
          let inlineStyle = row.settings.style;
          const voiceMatch = rawText.match(/^\[voice:([^\]]+)\]\s*/i);
          if (voiceMatch) {
            const requestedName = voiceMatch[1].trim().toLowerCase();
            const matchedVoice = library.voices.find(v => v.name.toLowerCase() === requestedName);
            const matchedSample = matchedVoice ? library.samples.find(s => s.voiceId === matchedVoice.id) : undefined;
            if (matchedSample) inlineVoiceSampleId = matchedSample.id;
            rawText = rawText.slice(voiceMatch[0].length);
          }
          const emotionMatch = rawText.match(/^\[emotion:([^\]]*)\]\s*/i);
          if (emotionMatch) {
            inlineStyle = emotionMatch[1].trim();
            rawText = rawText.slice(emotionMatch[0].length);
          }

          const sample = library.samples.find(value => value.id === inlineVoiceSampleId);
          const settings = {
            ...defaultSettings,
            ...row.settings,
            model: "orpheus_seedvc" as Model,
            studio_voice_id: row.settings.studio_voice_id || studioVoices[0]?.id || "",
            text: applyPronunciation(rawText, row.settings.pronunciation),
            sample: sample?.id ?? (inlineVoiceSampleId || null),
            transcript: sample?.transcript ?? "",
            language: row.settings.language || "",
            style: inlineStyle,
            embedding: sample?.embedding ?? null,
          };
          setProgressLabel(target.length > 1 ? `Block ${index + 1}/${target.length}…` : "Generating…");
          setNotice(target.length > 1 ? `Synthesizing block ${index + 1} of ${target.length}…` : "Synthesizing speech in your voice…");
          const output = await synthesize(launch.ws, settings, message => {
            setProgressLabel(message);
            setNotice(target.length > 1 ? `Block ${index + 1}/${target.length} · ${message}` : message);
          }, play);
          const take = { ...output, settings: { ...defaultSettings, ...row.settings }, blockId: row.id, created: Date.now() };
          await window.studio.take(take);
          setLibrary(data => ({ ...data, takes: [...data.takes, take] }));
          pickTake(row.id, take.id, index === target.length - 1);
        }
        setNotice(target.length > 1 ? `Generated ${target.length} blocks.` : "Speech generated.");
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
    const spokenBlocks = blocks.filter(row => row.settings.text.trim());
    const ids = master ? spokenBlocks.map(row => row.selected) : [selected?.id];
    if (!ids.length || ids.some(id => !id)) {
      setError(master ? "Generate audio for each block before exporting the master." : "Generate speech first to export.");
      return;
    }
    await task(async () => {
      setNotice(listen ? "Building master preview…" : `Saving ${format.toUpperCase()}…`);
      const takesList = ids as string[];
      const pause = block?.settings.period_pause ?? .36;
      const effect_preset = block?.settings.effect_preset ?? "broadcast";
      const pause_fill = block?.settings.pause_fill ?? "silence";
      const ranges = master ? spokenBlocks.flatMap(row => row.settings.timing ? [row.settings.timing] : []) : [];
      if (ranges.length && ranges.length !== spokenBlocks.length) {
        throw new Error("Every block in a timed project needs a start and end time.");
      }
      const timing = ranges.length ? ranges : undefined;
      const result = await window.studio.export({
        takes: takesList, timing, format, subtitles, ...(master ? {} : region),
        polish: master && !listen && blocks.some(row => row.settings.polish),
        pause, effect_preset, pause_fill,
      }, quick, listen);
      if (result.url) setPreview({ url: result.url, label: "Master preview", master: { takes: takesList, pause, timing, effect_preset, pause_fill } });
      setNotice(result.destination ? `Saved to ${result.destination}` : "");
    });
  }

  useEffect(() => {
    function handle(event: KeyboardEvent) {
      if (event.defaultPrevented || event.repeat) return;
      const typing = (event.target as HTMLElement | null)?.closest("textarea,input,select,[contenteditable],button");
      if (event.code === "Space" && !typing && !vault && !panel && player.current) {
        event.preventDefault(); player.current(); return;
      }
      if (!event.metaKey) return;
      const key = event.key.toLowerCase();
      if (!["enter", "r", "[", "]", "e"].includes(key)) return;
      event.preventDefault();
      if (busy || vault || panel) return;
      if (key === "enter") void generate(blocks.length > 1 || event.shiftKey);
      if (key === "r") setVault(true);
      if (key === "[") cycle(-1);
      if (key === "]") cycle(1);
      if (key === "e") void exportAudio("wav", false, true, undefined, true);
    }
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  });

  const model: Model = "orpheus_seedvc";
  const activeStudioVoiceId = block?.settings.studio_voice_id || studioVoices[0]?.id || "";
  const activeStudioVoice = studioVoices.find(v => v.id === activeStudioVoiceId) ?? studioVoices[0];
  const projectTakesCount = library.takes.filter(t => blocks.some(b => b.id === t.blockId)).length;

  return <main className="studio el-studio">
    <header className="topbar">
      <div className="brand">
        <img className="brand-logo" src="/youclone-logo.png" alt="YouClone" width={36} height={36} />
        <div>
          <strong>YouClone</strong>
          <span>Personal AI Voice Studio</span>
        </div>
      </div>
      <span className={launch?.ready ? "status online" : "status"}>
        <i />
        {launch?.ready
          ? (pvcTraining?.running
            ? `Training PVC · ${pvcTraining.percent}%`
            : activeStudioVoice
              ? `Ready · ${activeStudioVoice.name}`
              : "Ready · YouClone Studio (Orpheus-3B + Seed-VC 44kHz)")
          : launch?.serviceError ? "Engine offline" : "Warming up YouClone Studio…"}
      </span>
      <div className="topbar-actions">
        <button className="accent create-voice-btn btn-inline" disabled={!launch?.ready} onClick={() => setVoiceModal(true)}>
          {pvcTraining?.running ? (
            <><IconWaveform size={15} /><span>Training ({pvcTraining.percent}%)</span></>
          ) : (
            <><IconPlus size={15} /><span>Create Voice</span></>
          )}
        </button>
        <button className="toolbar-btn btn-inline" disabled={busy || !project} onClick={() => setPanel("history")}>
          <IconHistory size={15} />
          <span>History{projectTakesCount ? ` (${projectTakesCount})` : ""}</span>
        </button>
        <button className="toolbar-btn btn-inline" aria-pressed={showSettings} onClick={() => setShowSettings(v => !v)}>
          <IconSliders size={15} />
          <span>Voice Settings</span>
        </button>
        <div className="tools-dropdown" ref={menuRef}>
          <button className="toolbar-btn btn-inline" aria-expanded={menuOpen} onClick={() => setMenuOpen(v => !v)}>
            <span>Tools</span>
            <IconChevronDown size={14} className={menuOpen ? "chevron open" : "chevron"} />
          </button>
          {menuOpen && (
            <div className="tools-popover" role="menu">
              <button className="btn-inline" disabled={!launch?.ready} onClick={() => { setMenuOpen(false); setVoiceModal(true); }}>
                <IconWand size={15} /><span>Voice Cloning &amp; Character Design</span>
              </button>
              <button className="btn-inline" disabled={!launch?.ready || busy} onClick={() => { setMenuOpen(false); setPanel("changer"); }}>
                <IconMic size={15} /><span>Voice Changer (44.1 kHz)</span>
              </button>
              <button className="btn-inline" disabled={!launch?.ready || busy} onClick={() => { setMenuOpen(false); setPanel("sfx"); }}>
                <IconSoundwave size={15} /><span>Sound Effects</span>
              </button>
              <button className="btn-inline" disabled={!launch?.ready || busy} onClick={() => { setMenuOpen(false); setPanel("transcription"); }}>
                <IconDocument size={15} /><span>Transcribe &amp; Timed Dub</span>
              </button>
              <button className="btn-inline" disabled={busy} onClick={() => { setMenuOpen(false); setPanel("tools"); }}>
                <IconLab size={15} /><span>Voice Lab &amp; AI Director</span>
              </button>
              <div className="popover-divider" role="separator" />
              <button className="btn-inline" disabled={!project || busy} onClick={() => { setMenuOpen(false); void openProject(); }}>
                <IconFolderPlus size={15} /><span>New Project</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>

    <div className={showSettings ? "el-workbench with-drawer" : "el-workbench"}>
      <section className="el-canvas" aria-label="Text to Speech Composer">
        <div className="composer-card">
          {blocks.length > 1 && (
            <div className="block-tabs" role="tablist" aria-label="Timed script blocks">
              {blocks.map((row, idx) => (
                <button
                  key={row.id}
                  role="tab"
                  aria-selected={row.id === block?.id}
                  className={row.id === block?.id ? "tab-active" : ""}
                  onClick={() => focus(row.id)}
                >
                  Block {idx + 1}{row.settings.timing ? ` (${row.settings.timing.start.toFixed(1)}s)` : ""}
                </button>
              ))}
              <button
                disabled={!project || busy}
                onClick={() => {
                  if (!project) return;
                  const row = makeBlock(project.id);
                  if (block) row.settings = { ...defaultSettings, ...block.settings, text: "", timing: undefined };
                  setBlocks([...blocks, row]); focus(row.id);
                }}
              >+ Add</button>
            </div>
          )}

          {block && (
            <textarea
              ref={field}
              className="composer-textarea"
              disabled={busy}
              aria-label="Text to speech script"
              placeholder="Paste or type your script here… English or Hindi, any length — just click Generate Speech."
              value={block.settings.text}
              onChange={event => patch({ text: event.target.value }, block.id)}
            />
          )}

          <div className="composer-helpers">
            {!block?.settings.text.trim() ? (
              <div className="starter-bar" aria-label="Starter prompts">
                <span className="hint">Try sample:</span>
                {Object.entries(starterPrompts).map(([label, item]) => (
                  <button key={label} type="button" disabled={busy || !block} onClick={() => patch({ text: item.text, style: item.style })}>
                    {label}
                  </button>
                ))}
              </div>
            ) : (
              <div className="starter-bar">
                <button type="button" disabled={busy} onClick={() => tag("<sigh>")}>+ &lt;sigh&gt;</button>
                <button type="button" disabled={busy} onClick={() => tag("<chuckle>")}>+ &lt;chuckle&gt;</button>
                <button type="button" disabled={busy} onClick={() => tag("<laugh>")}>+ &lt;laugh&gt;</button>
                <button type="button" disabled={busy} onClick={() => tag("<gasp>")}>+ &lt;gasp&gt;</button>
                <button type="button" disabled={busy} onClick={() => tag("...")}>+ … pause</button>
                <button type="button" disabled={busy} onClick={() => tag("[pause 0.5s]")}>+ [pause 0.5s]</button>
                <button type="button" disabled={busy} onClick={() => patch({ text: "" })}>Clear text</button>
              </div>
            )}
          </div>

          <div className="composer-footer">
            <div className="composer-pickers">
              <label className="pill-select" title="Active Studio or Character Voice">
                <span
                  className={
                    activeStudioVoice?.mode === "professional"
                      ? "picker-icon pro"
                      : activeStudioVoice?.mode === "designed"
                        ? "picker-icon design"
                        : "picker-icon instant"
                  }
                >
                  {activeStudioVoice?.mode === "professional" ? (
                    <IconWaveform size={14} />
                  ) : activeStudioVoice?.mode === "designed" ? (
                    <IconWand size={14} />
                  ) : (
                    <IconBolt size={14} />
                  )}
                </span>
                <select
                  aria-label="Select voice"
                  disabled={busy || !block}
                  value={activeStudioVoiceId}
                  onChange={e => patch({ studio_voice_id: e.target.value })}
                >
                  {studioVoices.map(v => (
                    <option key={v.id} value={v.id}>
                      {v.name} · {v.mode === "professional" ? "PRO" : v.mode === "designed" ? "CHARACTER" : "INSTANT"}
                    </option>
                  ))}
                  {!studioVoices.length && <option value="">Studio Narrator · Leo</option>}
                </select>
              </label>

              <label className="pill-select" title="Speech Engine">
                <select
                  aria-label="Select model"
                  disabled={busy || !block}
                  value={model}
                  onChange={e => patch({ model: e.target.value as Model })}
                >
                  {Object.entries(models).map(([id, name]) => {
                    const info = modelsStatus.find(m => m.key === id);
                    const suffix = info && !info.cached ? " (Download)" : "";
                    return <option key={id} value={id}>{name}{suffix}</option>;
                  })}
                </select>
              </label>
            </div>

            <span className="script-stats" aria-label="Script telemetry">
              {scriptMetrics.words} words · {scriptMetrics.chars} chars · ~{scriptMetrics.clock}
            </span>

            <div className="composer-actions">
              {takes.length > 1 && (
                <div className="takes compact">
                  <button className="icon-btn sm" disabled={busy} aria-label="Previous take" onClick={() => cycle(-1)}>
                    <IconChevronLeft size={14} />
                  </button>
                  <span>Take {takeIndex + 1}/{takes.length}</span>
                  <button className="icon-btn sm" disabled={busy} aria-label="Next take" onClick={() => cycle(1)}>
                    <IconChevronRight size={14} />
                  </button>
                </div>
              )}
              <button
                className="accent generate-btn"
                disabled={busy || !launch?.ready || !block?.settings.text.trim()}
                onClick={() => void generate(blocks.length > 1)}
              >
                {busy ? (
                  progressLabel || "Generating…"
                ) : (
                  <>
                    <span>{blocks.length > 1 ? `Generate All (${blocks.length})` : "Generate Speech"}</span>
                    <kbd>⌘↵</kbd>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {(error || launch?.serviceError || launch?.modelWarning) && (
          <p role="alert" className="error">{error || launch?.serviceError || launch?.modelWarning}</p>
        )}
        {notice && <p aria-live="polite" className="notice">{notice}</p>}
      </section>

      {showSettings && block && (
        <aside className="inspector drawer" aria-label="Voice Settings">
          <div className="section-heading">
            <h2>Voice Settings</h2>
            <button className="icon-btn" onClick={() => setShowSettings(false)} aria-label="Close voice settings">
              <IconClose size={15} />
            </button>
          </div>
          <fieldset disabled={busy}>
            <label>Project
              <select className="project-picker" value={project?.id ?? ""} onChange={event => void openProject(event.target.value)}>
                {project && <option value={project.id}>{project.name || "Untitled project"}</option>}
                {library.projects.filter(item => item.id !== project?.id).map(item =>
                  <option key={item.id} value={item.id}>{item.name || "Untitled project"}</option>)}
              </select>
            </label>
            <label>Project title
              <input value={project?.name ?? ""} onChange={event => project && setProject({ ...project, name: event.target.value })} />
            </label>

            <label>Speed <output>{block.settings.speed.toFixed(2)}×</output>
              <input type="range" min=".7" max="1.3" step=".01" value={block.settings.speed}
                onChange={event => patch({ speed: event.target.valueAsNumber })} />
            </label>
            <label>Stability <output>{Math.round(block.settings.stability ?? 50)}%</output>
              <input type="range" min="0" max="100" step="1" value={block.settings.stability ?? 50}
                onChange={event => patch({ stability: event.target.valueAsNumber })} />
            </label>
            <label>Similarity <output>{Math.round(block.settings.similarity_boost ?? 75)}%</output>
              <input type="range" min="0" max="100" step="1" value={block.settings.similarity_boost ?? 75}
                onChange={event => patch({ similarity_boost: event.target.valueAsNumber })} />
            </label>
            <label>Style exaggeration <output>{Math.round(block.settings.style_exaggeration ?? 0)}%</output>
              <input type="range" min="0" max="100" step="1" value={block.settings.style_exaggeration ?? 0}
                onChange={event => patch({ style_exaggeration: event.target.valueAsNumber })} />
            </label>

            <label>Language override
              <input value={block.settings.language} placeholder="Auto (EN / HI)"
                onChange={event => patch({ language: event.target.value })} />
            </label>

            <label>Style instruction
              <textarea rows={2} value={block.settings.style}
                onChange={event => patch({ style: event.target.value })}
                placeholder="Optional delivery style (leave empty for pure voice clone)" />
            </label>

            <label>Mastering preset
              <select value={block.settings.effect_preset ?? "broadcast"} onChange={event => patch({ effect_preset: event.target.value as EffectPreset })}>
                <option value="broadcast">Broadcast — Leveler &amp; EQ</option>
                <option value="podcast">Podcast — Gated &amp; Tight</option>
                <option value="cinematic">Cinematic — Spacious</option>
                <option value="warm">Warm — Low-Mid Boost</option>
                <option value="bright">Bright — Airy Presence</option>
                <option value="raw">Raw — Unprocessed</option>
              </select>
            </label>
            <label>Pause fill
              <select value={block.settings.pause_fill ?? "silence"} onChange={event => patch({ pause_fill: event.target.value as PauseFill })}>
                <option value="silence">Digital silence (clean)</option>
                <option value="room_tone">Room tone (natural)</option>
              </select>
            </label>

            <details><summary>Audio Effects (Post-FX)</summary>
              <label>Voice filter
                <select value={block.settings.filter_mode ?? "none"} onChange={e => patch({ filter_mode: e.target.value as FilterMode })}>
                  <option value="none">None</option>
                  <option value="phone">Phone Line</option>
                  <option value="radio">Vintage Radio</option>
                  <option value="robot">Robot / Intercom</option>
                  <option value="megaphone">Megaphone</option>
                </select>
              </label>
              <label>Acoustic space
                <select value={block.settings.space_mode ?? "none"} onChange={e => patch({ space_mode: e.target.value as SpaceMode })}>
                  <option value="none">None (Dry Booth)</option>
                  <option value="room">Small Room</option>
                  <option value="hall">Concert Hall</option>
                  <option value="cathedral">Cathedral / Cavern</option>
                </select>
              </label>
              <label>Mic distance
                <select value={block.settings.distance_mode ?? "near"} onChange={e => patch({ distance_mode: e.target.value as DistanceMode })}>
                  <option value="near">Near (Close-Mic)</option>
                  <option value="medium">Medium</option>
                  <option value="far">Far (Room Perspective)</option>
                </select>
              </label>
              <label>Ambient bed
                <select value={block.settings.ambience_mode ?? "none"} onChange={e => patch({ ambience_mode: e.target.value as AmbienceMode })}>
                  <option value="none">None</option>
                  <option value="vinyl">Vinyl Crackle</option>
                  <option value="rain">Soft Rain</option>
                  <option value="tape">Analog Tape Hiss</option>
                  <option value="cafe">Café Murmur</option>
                </select>
              </label>
            </details>

            <details><summary>Advanced</summary>
              <label>Pronunciation aliases
                <textarea rows={3} value={block.settings.pronunciation ?? ""}
                  placeholder={"GTA = G T A\n1757 = सत्रह सौ सत्तावन"}
                  onChange={event => patch({ pronunciation: event.target.value })} />
              </label>
              <label>Temperature <output>{block.settings.temperature.toFixed(2)}</output>
                <input type="range" min="0" max="2" step=".05" value={block.settings.temperature}
                  onChange={event => patch({ temperature: event.target.valueAsNumber })} />
              </label>
              <label>Seed<input type="number" min="0" max="4294967295" value={block.settings.seed}
                onChange={event => patch({ seed: event.target.valueAsNumber || 0 })} /></label>
              <label>Comma pause (s)<input type="number" min="0" max="3" step=".05" value={block.settings.comma_pause}
                onChange={event => patch({ comma_pause: event.target.valueAsNumber || 0 })} /></label>
              <label>Sentence pause (s)<input type="number" min="0" max="3" step=".05" value={block.settings.period_pause}
                onChange={event => patch({ period_pause: event.target.valueAsNumber || 0 })} /></label>
              <label className="check"><input type="checkbox" checked={block.settings.trim_silence ?? true}
                onChange={event => patch({ trim_silence: event.target.checked })} /> Trim edge silence (−50 dBFS)</label>
              <label className="check"><input type="checkbox" checked={block.settings.watermark ?? true}
                onChange={event => patch({ watermark: event.target.checked })} /> AI provenance watermark</label>
            </details>

            <label className="check"><input type="checkbox" checked={block.settings.speaker_boost ?? true}
              onChange={event => patch({ speaker_boost: event.target.checked })} /> Speaker Boost</label>
            <label className="check"><input type="checkbox" checked={block.settings.polish}
              onChange={event => patch({ polish: event.target.checked })} /> Studio Polish</label>
            <label className="check"><input type="checkbox" checked={audition}
              onChange={event => setAudition(event.target.checked)} /> Audition live chunks while generating</label>
          </fieldset>
        </aside>
      )}
    </div>

    <Timeline
      key={audio}
      audio={audio}
      autoPlay={autoPlayNext}
      label={preview?.label ?? (selected ? `${activeStudioVoice?.name ?? "Studio Voice"} · Take ${takeIndex + 1}` : "Generated speech will appear here")}
      locked={busy || !launch?.ready}
      canExportTake={preview ? !!preview.master : !!selected}
      canExportMaster={blocks.length > 1 && blocks.some(row => !!row.settings.text.trim()) &&
        blocks.every(row => !row.settings.text.trim() || !!row.selected)}
      setPlayer={setPlayer}
      exportAudio={(format, subtitles, master, region) => {
        if (!master && preview?.master) {
          const masterSettings = preview.master;
          void task(async () => {
            const result = await window.studio.export({ ...masterSettings, format, subtitles, ...region }, false, false);
            if (result.destination) setNotice(`Saved to ${result.destination}`);
          });
          return;
        }
        void exportAudio(format, subtitles, master, region);
      }}
    />

    {panel === "transcription" && <Transcription close={() => setPanel(undefined)} sources={library.sources}
      importText={text => { appendScript(text); setPanel(undefined); }}
      importCaptions={captions => { setPanel(undefined); void openProject(undefined, captions); }} />}
    {panel === "changer" && <VoiceChanger close={() => setPanel(undefined)} sources={library.sources}
      voices={library.voices} samples={library.samples}
      attachTake={attachExternalTake}
      exportTake={async (takeId, format) => { await window.studio.export({ takes: [takeId], format }, false, false); }} />}
    {panel === "sfx" && <SoundEffectsStudio close={() => setPanel(undefined)}
      attachTake={attachExternalTake}
      exportTake={async (takeId, format) => { await window.studio.export({ takes: [takeId], format }, false, false); }} />}
    {(panel === "tools" || panel === "lora") && <StudioToolsModal
      key={panel}
      initialTab={panel === "lora" ? "lora" : "compare"}
      close={() => setPanel(undefined)}
      sources={library.sources} voices={library.voices} samples={library.samples}
      adapters={adapters}
      modelsStatus={modelsStatus}
      wsEndpoint={launch?.ws} ready={!!launch?.ready}
      activeText={block?.settings.text ?? ""}
      applyDirectionToBlock={patchObj => patch(patchObj)}
      applySpeedToBlock={speed => patch({ speed })}
      attachTake={attachExternalTake}
      onRefreshModels={refreshModelsStatus}
      onLoraTrained={async adapter => {
        const { voice, sample } = await window.studio.sample(adapter.name, {
          id: adapter.sample_id,
          style: "LoRA Fine-Tuned",
          original: adapter.sources.join(", "),
          transcript: adapter.sample_transcript,
          language: adapter.sample_language,
          seconds: adapter.sample_seconds,
          embedding: null,
        });
        setLibrary(data => ({
          ...data,
          voices: data.voices.some(item => item.id === voice.id) ? data.voices : [...data.voices, voice],
          samples: [...data.samples, sample],
        }));
        patch({
          model: "orpheus_seedvc",
          sample: sample.id,
          lora_adapter: adapter.id,
        });
        setNotice(`Activated voice profile "${adapter.name}".`);
      }}
      onActivateLora={adapter => {
        const matched = library.samples.find(s => s.id === adapter.sample_id);
        patch({
          model: "orpheus_seedvc",
          ...(matched ? { sample: matched.id } : {}),
          lora_adapter: adapter.id,
        });
        setNotice(`Activated voice profile "${adapter.name}".`);
      }}
    />}
    {voiceModal && <VoiceCreationModal
      voices={studioVoices}
      activeVoiceId={activeStudioVoiceId}
      training={pvcTraining}
      ready={!!launch?.ready}
      close={() => setVoiceModal(false)}
      onSelectVoice={voice => {
        patch({ studio_voice_id: voice.id });
        setNotice(`Switched active voice to "${voice.name}".`);
      }}
      onRefresh={refreshModelsStatus}
    />}
    {panel === "history" && <TakeHistory close={() => setPanel(undefined)}
      takes={library.takes.filter(take => blocks.some(row => row.id === take.blockId))}
      select={take => { focus(take.blockId); pickTake(take.blockId, take.id, false); setPanel(undefined); }}
      save={async (take, format) => { await window.studio.export({ takes: [take.id], format }, false, false); }} />}
    {vault && <Vault sources={library.sources} voices={library.voices} ready={!!launch?.ready} close={() => setVault(false)}
      saved={(voice, sample) => {
        setLibrary(data => ({ ...data, voices: data.voices.some(item => item.id === voice.id) ? data.voices : [...data.voices, voice],
          samples: [...data.samples, sample] }));
        patch({ sample: sample.id });
        setPreview({ url: `/media/samples/${sample.id}.wav`, label: `${voice.name} · ${sample.style}` });
        setVault(false);
      }} />}
  </main>;
}
