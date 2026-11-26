"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  IconCheckCircle,
  IconClock,
  IconClose,
  IconDirector,
  IconDownload,
  IconFilter,
  IconInfoCircle,
  IconLayers,
  IconMic,
  IconShield,
  IconShuffle,
  IconSoundwave,
  IconStop,
  IconSwap,
} from "./icons";
import {
  defaultSettings,
  describe,
  type EffectPreset,
  type Format,
  type LoraAdapter,
  type Model,
  type ModelStatusInfo,
  type Sample,
  type Settings,
  type Take,
  type Voice,
} from "./types";

export function Panel({ title, subtitle, close, children }: { title: string; subtitle?: string; close(): void; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog ref={dialog} aria-label={title} onCancel={event => { event.preventDefault(); close(); }}>
    <div className="modal-heading">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      <button type="button" className="icon-btn" onClick={close} aria-label={`Close ${title}`}>
        <IconClose size={15} />
      </button>
    </div>
    {children}
  </dialog>;
}

export type Caption = { start: number; end: number; text: string };
type Transcript = { text: string; language: string; segments: Caption[] };

function srtTime(seconds: number) {
  const ms = Math.round(Math.max(0, seconds) * 1000);
  return [Math.floor(ms / 3600000), Math.floor(ms / 60000) % 60, Math.floor(ms / 1000) % 60]
    .map(value => String(value).padStart(2, "0")).join(":") + "," + String(ms % 1000).padStart(3, "0");
}

function download(name: string, contents: string) {
  const url = URL.createObjectURL(new Blob([contents], { type: "text/plain;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export function Transcription({ close, sources, importText, importCaptions }: {
  close(): void; sources: string[]; importText(text: string): void; importCaptions(captions: Caption[]): void;
}) {
  const [source, setSource] = useState("");
  const [file, setFile] = useState<File>();
  const [result, setResult] = useState<Transcript>();
  const [script, setScript] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
  const [openingMic, setOpeningMic] = useState(false);
  const resources = useRef<{ stream?: MediaStream; recorder?: MediaRecorder }>({});
  const mounted = useRef(false);
  const request = useRef<AbortController | null>(null);
  useEffect(() => {
    mounted.current = true;
    const capture = resources.current;
    return () => {
      mounted.current = false;
      request.current?.abort();
      capture.stream?.getTracks().forEach(track => track.stop());
    };
  }, []);

  async function dictate() {
    const capture = resources.current;
    if (capture.recorder?.state === "recording") { capture.recorder.stop(); return; }
    setError("");
    setOpeningMic(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mounted.current) { stream.getTracks().forEach(track => track.stop()); return; }
      capture.stream = stream;
      const recorder = new MediaRecorder(stream);
      capture.recorder = recorder;
      const chunks: Blob[] = [];
      recorder.ondataavailable = event => chunks.push(event.data);
      recorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        capture.recorder = undefined;
        if (!mounted.current) return;
        setRecording(false);
        const recordingFile = new File(chunks, "Dictation.webm", { type: recorder.mimeType });
        setSource(""); setFile(recordingFile);
        void convert(recordingFile);
      };
      recorder.start();
      setRecording(true);
    } catch (problem) {
      capture.stream?.getTracks().forEach(track => track.stop());
      if (mounted.current) setError(describe(problem));
    } finally {
      if (mounted.current) setOpeningMic(false);
    }
  }

  async function convert(input = file) {
    const controller = new AbortController();
    request.current = controller;
    setBusy(true); setError(""); setResult(undefined); setScript("");
    try {
      const body = new FormData();
      if (input) body.set("file", input); else body.set("source", source);
      const response = await fetch("/api/transcribe", { method: "POST", body, signal: controller.signal });
      const value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Transcription failed.");
      if (!value.text.trim()) throw new Error("No speech was detected.");
      setResult(value); setScript(value.segments.map((caption: Caption) => caption.text).join("\n\n") || value.text);
    } catch (problem) {
      if (!controller.signal.aborted) setError(describe(problem));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return <Panel title="Transcription & Timed Dubbing" subtitle="Transcribe audio/video locally with Whisper, dictate live, or create a timed dubbing project." close={close}>
    <fieldset disabled={busy || recording || openingMic}>
      <label>Audio or video<input type="file" accept="audio/*,video/*,.mkv,.webm"
        onChange={event => { setFile(event.target.files?.[0]); setSource(""); }} /></label>
      {!!sources.length && <label>Workspace recording<select value={source} onChange={event => { setSource(event.target.value); setFile(undefined); }}>
        <option value="">Choose a recording</option>{sources.map(item => <option key={item}>{item}</option>)}
      </select></label>}
    </fieldset>
    <div className="tool-actions">
      <button className="accent" disabled={busy || recording || openingMic || (!file && !source)} onClick={() => void convert()}>Transcribe {file?.name ?? source}</button>
      <button className="btn-inline" disabled={busy || openingMic} onClick={() => void dictate()}>
        {recording ? <><IconStop size={13} /><span>Stop &amp; Transcribe</span></> : <><IconMic size={14} /><span>Live Dictate</span></>}
      </button>
    </div>
    <p aria-live="polite">{busy ? "Transcribing locally…" : recording ? "Recording microphone…" : result ? `Detected language: ${result.language.toUpperCase()} · ${result.segments.length} timed segments` : ""}</p>
    {error && <p role="alert" className="error">{error}</p>}
    {result && <>
      <label>Transcript<textarea rows={9} value={script} onChange={event => setScript(event.target.value)} /></label>
      <div className="tool-actions">
        <button onClick={() => download("transcript.txt", script)}>Download text</button>
        <button disabled={!result.segments.length} onClick={() => download("captions.srt", result.segments.map((caption, i) =>
          `${i + 1}\n${srtTime(caption.start)} --> ${srtTime(caption.end)}\n${caption.text}\n`).join("\n"))}>Download original captions</button>
        <button disabled={!result.segments.length} onClick={() => importCaptions(result.segments)}>New timed project</button>
        <button className="accent" disabled={!script.trim()} onClick={() => importText(script)}>Add to script</button>
      </div>
    </>}
  </Panel>;
}

export function VoiceChanger({
  close,
  sources,
  voices,
  samples,
  attachTake,
  exportTake,
}: {
  close(): void;
  sources: string[];
  voices: Voice[];
  samples: Sample[];
  attachTake(takeId: string, seconds: number, text: string, settingsPatch: Partial<Settings>): Promise<void>;
  exportTake(takeId: string, format: Format): Promise<void>;
}) {
  const [source, setSource] = useState(sources[0] ?? "");
  const [file, setFile] = useState<File>();
  const [model, setModel] = useState<Model>("orpheus_seedvc");
  const [sampleId, setSampleId] = useState(samples[0]?.id ?? "");
  const [style, setStyle] = useState("Natural conversational delivery matching original pacing.");
  const [matchTiming, setMatchTiming] = useState(true);
  const [preset, setPreset] = useState<EffectPreset>("broadcast");
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ id: string; seconds: number; source_seconds: number; text: string; language: string }>();
  const resources = useRef<{ stream?: MediaStream; recorder?: MediaRecorder }>({});

  useEffect(() => () => {
    resources.current.stream?.getTracks().forEach(t => t.stop());
  }, []);

  async function toggleMic() {
    if (recording) {
      resources.current.recorder?.stop();
      return;
    }
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      resources.current.stream = stream;
      const recorder = new MediaRecorder(stream);
      resources.current.recorder = recorder;
      const chunks: Blob[] = [];
      recorder.ondataavailable = e => chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        setRecording(false);
        setSource("");
        setFile(new File(chunks, "VoiceChangerInput.webm", { type: recorder.mimeType }));
      };
      recorder.start();
      setRecording(true);
    } catch (e) {
      setError(describe(e));
    }
  }

  async function runConvert() {
    setBusy(true); setError(""); setResult(undefined);
    try {
      const chosenSample = samples.find(s => s.id === sampleId);
      const body = new FormData();
      if (file) body.set("file", file); else body.set("source", source);
      body.set("model", model);
      body.set("studio_voice_id", "");
      if (chosenSample) {
        body.set("sample", chosenSample.id);
        body.set("transcript", chosenSample.transcript);
        body.set("language", chosenSample.language);
      }
      body.set("style", style);
      body.set("match_timing", String(matchTiming));
      body.set("effect_preset", preset);
      const res = await fetch("/api/convert", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Voice conversion failed.");
      setResult(data);
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  return <Panel
    title="Speech-to-Speech Voice Changer"
    subtitle="Convert any recording into a cloned or designed voice while matching original timing."
    close={close}
  >
    <div className="modal-grid">
      <section>
        <label>Upload source audio / video
          <input type="file" disabled={busy || recording} accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.webm"
            onChange={e => { setFile(e.target.files?.[0]); setSource(""); }} />
        </label>
        {!!sources.length && <label>Or workspace recording
          <select disabled={busy || recording} value={source} onChange={e => { setSource(e.target.value); setFile(undefined); }}>
            <option value="">Choose a recording</option>
            {sources.map(item => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>}
        <div className="tool-actions">
          <button className="btn-inline" disabled={busy} onClick={() => void toggleMic()}>
            {recording ? <><IconStop size={13} /><span>Stop Recording</span></> : <><IconMic size={14} /><span>Record Source Clip</span></>}
          </button>
          {file && <span className="hint">{file.name}</span>}
        </div>
      </section>
      <section>
        <label>Synthesis model
          <select disabled={busy} value={model} onChange={e => setModel(e.target.value as Model)}>
            <option value="orpheus_seedvc">YouClone Studio (Seed-VC 44.1 kHz)</option>
          </select>
        </label>
        <label>Target voice sample
          <select disabled={busy} value={sampleId} onChange={e => setSampleId(e.target.value)}>
            <option value="">Select a saved voice</option>
            {voices.map(v => <optgroup key={v.id} label={v.name}>
              {samples.filter(s => s.voiceId === v.id).map(s =>
                <option key={s.id} value={s.id}>{s.style} · {s.seconds.toFixed(1)} s</option>)}
            </optgroup>)}
          </select>
        </label>
        <label>Mastering preset
          <select disabled={busy} value={preset} onChange={e => setPreset(e.target.value as EffectPreset)}>
            <option value="broadcast">Broadcast</option>
            <option value="podcast">Podcast</option>
            <option value="cinematic">Cinematic</option>
            <option value="warm">Warm</option>
            <option value="bright">Bright</option>
            <option value="raw">Raw</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={matchTiming} disabled={busy} onChange={e => setMatchTiming(e.target.checked)} />
          Match source duration (pitch-locked atempo)
        </label>
      </section>
    </div>
    <div className="tool-actions">
      <button className="accent btn-inline" disabled={busy || recording || (!file && !source)} onClick={() => void runConvert()}>
        <IconSwap size={14} />
        <span>{busy ? "Converting voice…" : "Convert Voice"}</span>
      </button>
    </div>
    {error && <p role="alert" className="error">{error}</p>}
    {result && <div className="prepared">
      <p className="hint">
        Transcribed ({result.language.toUpperCase()}): “{result.text}” · Source {result.source_seconds}s → Converted {result.seconds.toFixed(2)}s
      </p>
      <audio controls src={`/media/takes/${result.id}.wav`} />
      <div className="tool-actions">
        <button className="accent" onClick={() => void attachTake(result.id, result.seconds, result.text, {
          model, sample: sampleId, style, effect_preset: preset,
        }).then(close)}>
          Use in focused block
        </button>
        <button className="btn-inline" onClick={() => void exportTake(result.id, "wav")}>
          <IconDownload size={14} /><span>Download WAV</span>
        </button>
      </div>
    </div>}
  </Panel>;
}

const SFX_PRESETS = [
  { label: "Rain & Thunder", prompt: "Heavy rain downpour with distant rolling thunder rumble" },
  { label: "Cinematic Riser", prompt: "Suspenseful cinematic riser swell transition" },
  { label: "Sci-Fi Drone", prompt: "Deep sci-fi spacecraft ambient hull drone hum" },
  { label: "Vinyl Crackle", prompt: "Warm vintage vinyl record crackle and campfire texture" },
  { label: "Café Ambience", prompt: "Cozy busy café crowd murmur and room ambience" },
  { label: "Heartbeat", prompt: "Tense rhythmic human heartbeat pulse" },
  { label: "Mechanical Typing", prompt: "Fast mechanical keyboard typing clicks" },
  { label: "Mountain Wind", prompt: "Howling mountain wind gale whoosh" },
  { label: "Radio Static", prompt: "Vintage shortwave radio static tuning interference" },
];

export function SoundEffectsStudio({
  close,
  attachTake,
  exportTake,
}: {
  close(): void;
  attachTake(takeId: string, seconds: number, text: string, settingsPatch: Partial<Settings>): Promise<void>;
  exportTake(takeId: string, format: Format): Promise<void>;
}) {
  const [prompt, setPrompt] = useState(SFX_PRESETS[0].prompt);
  const [duration, setDuration] = useState(6);
  const [seed, setSeed] = useState(42);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ id: string; seconds: number; prompt: string; seed: number }>();

  async function generateSfx() {
    setBusy(true); setError(""); setResult(undefined);
    try {
      const res = await fetch("/api/sfx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, duration, seed }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "SFX synthesis failed.");
      setResult(data);
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  return <Panel
    title="Sound Effects & Ambience Synthesizer"
    subtitle="Generate procedural soundscapes, foley, and ambient beds for dialogue & audiobook production."
    close={close}
  >
    <div className="tag-row">
      {SFX_PRESETS.map(item => (
        <button key={item.label} type="button" disabled={busy} onClick={() => setPrompt(item.prompt)}>
          {item.label}
        </button>
      ))}
    </div>
    <fieldset disabled={busy}>
      <label>Sound effect description
        <textarea rows={3} value={prompt} onChange={e => setPrompt(e.target.value)}
          placeholder="Describe the sound effect or ambient texture…" />
      </label>
      <div className="clip-range">
        <label>Duration ({duration}s)
          <input type="range" min={1} max={30} step={1} value={duration}
            onChange={e => setDuration(e.target.valueAsNumber)} />
        </label>
        <label>Seed
          <input type="number" min={0} max={2147483647} value={seed}
            onChange={e => setSeed(e.target.valueAsNumber || 0)} />
        </label>
      </div>
    </fieldset>
    <div className="tool-actions">
      <button className="accent btn-inline" disabled={busy || !prompt.trim()} onClick={() => void generateSfx()}>
        <IconSoundwave size={14} />
        <span>{busy ? "Synthesizing soundscape…" : "Generate Sound Effect"}</span>
      </button>
      <button className="btn-inline" disabled={busy} onClick={() => setSeed(Math.floor(Math.random() * 100000))}>
        <IconShuffle size={14} />
        <span>Randomize Seed</span>
      </button>
    </div>
    {error && <p role="alert" className="error">{error}</p>}
    {result && <div className="prepared">
      <p className="hint">Generated {result.seconds.toFixed(1)}s SFX · Seed {result.seed} · Watermarked (YC)</p>
      <audio controls src={`/media/takes/${result.id}.wav`} />
      <div className="tool-actions">
        <button className="accent" onClick={() => void attachTake(result.id, result.seconds, `[SFX: ${result.prompt}]`, {}).then(close)}>
          Insert into focused block
        </button>
        <button className="btn-inline" onClick={() => void exportTake(result.id, "wav")}>
          <IconDownload size={14} /><span>Download WAV</span>
        </button>
      </div>
    </div>}
  </Panel>;
}

export function StudioToolsModal({
  close,
  initialTab = "compare",
  sources,
  voices,
  samples,
  adapters,
  modelsStatus,
  wsEndpoint,
  ready,
  activeText,
  applyDirectionToBlock,
  applySpeedToBlock,
  attachTake,
  onLoraTrained,
  onActivateLora,
  onRefreshModels,
}: {
  close(): void;
  initialTab?: "lora" | "compare" | "isolate" | "director" | "watermark";
  sources: string[];
  voices: Voice[];
  samples: Sample[];
  adapters: LoraAdapter[];
  modelsStatus: ModelStatusInfo[];
  wsEndpoint?: string;
  ready: boolean;
  activeText: string;
  applyDirectionToBlock(patch: Partial<Settings>): void;
  applySpeedToBlock(speed: number): void;
  attachTake(takeId: string, seconds: number, text: string, settingsPatch: Partial<Settings>): Promise<void>;
  onLoraTrained(adapter: LoraAdapter): Promise<void>;
  onActivateLora(adapter: LoraAdapter): void;
  onRefreshModels(): Promise<void>;
}) {
  const [tab, setTab] = useState<"lora" | "compare" | "isolate" | "director" | "watermark">(initialTab);

  // LoRA Fine-Tuning & Model Checkpoint Manager state
  const [adapterName, setAdapterName] = useState("My Fine-Tuned Voice");
  const [selectedSources, setSelectedSources] = useState<string[]>(sources);
  const [rank, setRank] = useState(8);
  const [steps, setSteps] = useState(60);
  const [learningRate, setLearningRate] = useState(1e-4);
  const [loraBusy, setLoraBusy] = useState(false);
  const [loraProgress, setLoraProgress] = useState("");
  const [loraError, setLoraError] = useState("");
  const [downloadingModel, setDownloadingModel] = useState<Model | null>(null);
  const [modelError, setModelError] = useState("");

  // A/B Compare state
  const [phrase, setPhrase] = useState(
    activeText.trim() || "The quick brown fox jumps over the lazy dog. Can you hear the difference in tone and clarity?"
  );
  const [voiceA, setVoiceA] = useState(samples[0]?.id ?? "qwen:Conversational");
  const [voiceB, setVoiceB] = useState(samples[1]?.id ?? "qwen:Documentary");
  const [compareBusy, setCompareBusy] = useState(false);
  const [compareError, setCompareError] = useState("");
  const [urlA, setUrlA] = useState("");
  const [urlB, setUrlB] = useState("");

  // Voice Isolator state
  const [isoSource, setIsoSource] = useState(sources[0] ?? "");
  const [isoFile, setIsoFile] = useState<File>();
  const [isoBusy, setIsoBusy] = useState(false);
  const [isoError, setIsoError] = useState("");
  const [isoResult, setIsoResult] = useState<{ id: string; seconds: number }>();

  // Directorial AI & Rate-Fit state
  const [directionInput, setDirectionInput] = useState("Urgent and surprised, whispered close to the mic");
  const [directionResult, setDirectionResult] = useState<{
    tokens: Record<string, string[]>;
    instruct_prompt: string;
    translate_hint: string;
    rate_bias: number;
    recommended_stability: number;
    recommended_style: number;
  }>();
  const [slotText, setSlotText] = useState(activeText.trim() || "We have less than ten seconds before the vault doors lock from the inside.");
  const [slotSeconds, setSlotSeconds] = useState(3.5);
  const [rateResult, setRateResult] = useState<{
    word_count: number;
    char_count: number;
    slot_seconds: number;
    estimated_seconds: number;
    required_speed: number;
    recommended_speed: number;
    verdict: string;
    suggestion: string;
  }>();
  const [toolError, setToolError] = useState("");

  // Watermark Detector state
  const [wmSource, setWmSource] = useState(sources[0] ?? "");
  const [wmFile, setWmFile] = useState<File>();
  const [wmBusy, setWmBusy] = useState(false);
  const [wmError, setWmError] = useState("");
  const [wmResult, setWmResult] = useState<{
    detected: boolean;
    confidence: number;
    bits_matched?: string;
    signature: string;
    method?: string;
    details: string;
  }>();

  async function synthOne(choice: string): Promise<string> {
    if (!wsEndpoint) throw new Error("Local inference service is not connected.");
    const isPreset = choice.startsWith("qwen:");
    const chosenSample = !isPreset ? samples.find(s => s.id === choice) : undefined;
    const payload = {
      ...defaultSettings,
      model: "orpheus_seedvc",
      text: phrase,
      sample: chosenSample?.id ?? null,
      transcript: chosenSample?.transcript ?? "",
      language: chosenSample?.language ?? "",
      style: isPreset ? choice.slice(5) + " delivery with natural articulation." : "",
    };
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsEndpoint);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => socket.send(JSON.stringify(payload));
      socket.onmessage = ({ data }) => {
        if (typeof data !== "string") return;
        const msg = JSON.parse(data);
        if (msg.type === "done") { resolve(`/media/takes/${msg.id}.wav`); socket.close(); }
        if (msg.type === "error") { reject(new Error(msg.message)); socket.close(); }
      };
      socket.onerror = () => reject(new Error("WebSocket synthesis error."));
    });
  }

  async function runCompare() {
    setCompareBusy(true); setCompareError(""); setUrlA(""); setUrlB("");
    try {
      const first = await synthOne(voiceA);
      setUrlA(first);
      const second = await synthOne(voiceB);
      setUrlB(second);
    } catch (e) {
      setCompareError(describe(e));
    } finally {
      setCompareBusy(false);
    }
  }

  async function runIsolate() {
    setIsoBusy(true); setIsoError(""); setIsoResult(undefined);
    try {
      const body = new FormData();
      if (isoFile) body.set("file", isoFile); else body.set("source", isoSource);
      const res = await fetch("/api/isolate", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Voice isolation failed.");
      setIsoResult(data);
    } catch (e) {
      setIsoError(describe(e));
    } finally {
      setIsoBusy(false);
    }
  }

  async function runDirection() {
    setToolError("");
    try {
      const res = await fetch("/api/tools/direction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: directionInput }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error("Could not parse direction.");
      setDirectionResult(data);
    } catch (e) {
      setToolError(describe(e));
    }
  }

  async function runRateFit() {
    setToolError("");
    try {
      const res = await fetch("/api/tools/rate-fit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: slotText, slot_seconds: slotSeconds, language: "en" }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error("Could not calculate slot fit.");
      setRateResult(data);
    } catch (e) {
      setToolError(describe(e));
    }
  }

  async function runWatermarkCheck() {
    setWmBusy(true); setWmError(""); setWmResult(undefined);
    try {
      const body = new FormData();
      if (wmFile) body.set("file", wmFile); else body.set("source", wmSource);
      const res = await fetch("/api/watermark/detect", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Watermark check failed.");
      setWmResult(data);
    } catch (e) {
      setWmError(describe(e));
    } finally {
      setWmBusy(false);
    }
  }

  async function runTrainLora() {
    if (!selectedSources.length || !adapterName.trim()) return;
    setLoraBusy(true);
    setLoraError("");
    setLoraProgress("Preparing workspace recordings…");
    const timer = window.setInterval(async () => {
      try {
        const res = await fetch("/api/lora/status");
        if (!res.ok) return;
        const st = await res.json();
        if (st.message) setLoraProgress(st.message);
      } catch {}
    }, 800);
    try {
      const res = await fetch("/api/lora/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: adapterName.trim(),
          sources: selectedSources,
          rank,
          steps,
          learning_rate: learningRate,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "LoRA fine-tuning failed.");
      await onLoraTrained(data.adapter);
      await onRefreshModels();
      setLoraProgress(`Trained ${data.adapter.name} · Final loss ${data.adapter.final_loss}`);
    } catch (e) {
      setLoraError(describe(e));
    } finally {
      window.clearInterval(timer);
      setLoraBusy(false);
    }
  }

  async function runDownloadModel(key: Model) {
    setDownloadingModel(key);
    setModelError("");
    try {
      const res = await fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: key }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Checkpoint download failed.");
      await onRefreshModels();
    } catch (e) {
      setModelError(describe(e));
    } finally {
      setDownloadingModel(null);
    }
  }

  return <Panel
    title="Voice Lab & Studio Tools"
    subtitle="LoRA Voice Fine-Tuning, 48 kHz Model Checkpoints, A/B Voice Comparison, Vocal Isolation, Directorial AI & Watermark Verification."
    close={close}
  >
    <div className="tag-row">
      <button className="btn-inline" aria-pressed={tab === "lora"} onClick={() => setTab("lora")}>
        <IconLayers size={14} /><span>Fine-Tune &amp; Models</span>
      </button>
      <button className="btn-inline" aria-pressed={tab === "compare"} onClick={() => setTab("compare")}>
        <IconSwap size={14} /><span>A/B Voice Compare</span>
      </button>
      <button className="btn-inline" aria-pressed={tab === "isolate"} onClick={() => setTab("isolate")}>
        <IconFilter size={14} /><span>Voice Isolator</span>
      </button>
      <button className="btn-inline" aria-pressed={tab === "director"} onClick={() => setTab("director")}>
        <IconDirector size={14} /><span>Directorial AI &amp; Slot Fit</span>
      </button>
      <button className="btn-inline" aria-pressed={tab === "watermark"} onClick={() => setTab("watermark")}>
        <IconShield size={14} /><span>Watermark Detector</span>
      </button>
    </div>

    {tab === "lora" && <section className="prepared">
      <h3>Train Personal Voice LoRA (OmniVoice MLX)</h3>
      <p className="hint">Slices clean speech from your workspace recordings, transcribes with Whisper, and trains an Apple Silicon MLX LoRA adapter across OmniVoice’s attention &amp; 8 codebook heads.</p>
      <div className="modal-grid">
        <div>
          <label>Voice / Adapter Name
            <input value={adapterName} disabled={loraBusy} onChange={e => setAdapterName(e.target.value)}
              placeholder="My Fine-Tuned Voice" />
          </label>
          <p className="hint">Workspace training recordings:</p>
          <div className="tool-actions">
            {sources.map(src => (
              <label key={src} className="check">
                <input
                  type="checkbox"
                  disabled={loraBusy}
                  checked={selectedSources.includes(src)}
                  onChange={e => setSelectedSources(curr =>
                    e.target.checked ? [...curr, src] : curr.filter(item => item !== src)
                  )}
                />
                {src}
              </label>
            ))}
            {!sources.length && <span className="hint">No workspace recordings found.</span>}
          </div>
        </div>
        <div>
          <div className="clip-range">
            <label>LoRA Rank
              <select value={rank} disabled={loraBusy} onChange={e => setRank(Number(e.target.value))}>
                <option value={8}>Rank 8 (Fast &amp; Clean)</option>
                <option value={16}>Rank 16 (High Capacity)</option>
              </select>
            </label>
            <label>Training Steps
              <select value={steps} disabled={loraBusy} onChange={e => setSteps(Number(e.target.value))}>
                <option value={40}>40 steps (~1 min)</option>
                <option value={60}>60 steps (Recommended)</option>
                <option value={100}>100 steps (Deep Fit)</option>
              </select>
            </label>
          </div>
          <label>Learning Rate
            <select value={String(learningRate)} disabled={loraBusy} onChange={e => setLearningRate(Number(e.target.value))}>
              <option value="0.0001">1e-4 (Balanced)</option>
              <option value="0.0002">2e-4 (Faster Adaptation)</option>
              <option value="0.00005">5e-5 (Gentle Polish)</option>
            </select>
          </label>
        </div>
      </div>
      <div className="tool-actions">
        <button
          className="accent btn-inline"
          disabled={!ready || loraBusy || !selectedSources.length || !adapterName.trim()}
          onClick={() => void runTrainLora()}
        >
          <IconLayers size={14} />
          <span>{loraBusy ? "Training Personal LoRA…" : "Train Personal Voice LoRA"}</span>
        </button>
        {loraProgress && <span className="notice" aria-live="polite">{loraProgress}</span>}
      </div>
      {loraError && <p role="alert" className="error">{loraError}</p>}

      {!!adapters.length && <div className="prepared">
        <h3>Trained Personal Voice Adapters ({adapters.length})</h3>
        <div className="take-history">
          {adapters.map(ad => (
            <article key={ad.id}>
              <p><strong>{ad.name}</strong></p>
              <small>
                Rank {ad.rank} · {ad.steps} steps · {ad.clips} clips · Loss {ad.final_loss} · Sources: {ad.sources.join(", ")}
              </small>
              <div className="tool-actions">
                <button className="accent" onClick={() => { onActivateLora(ad); close(); }}>
                  Activate on focused block
                </button>
              </div>
            </article>
          ))}
        </div>
      </div>}

      <div className="prepared">
        <h3>Model Checkpoint Manager</h3>
        <p className="hint">Manage locally cached Apple Silicon synthesis models in your `.voxm5/huggingface` directory.</p>
        {modelError && <p role="alert" className="error">{modelError}</p>}
        <div className="take-history">
          {modelsStatus.map(m => (
            <article key={m.key}>
              <p><strong>{m.label}</strong> ({(m.sample_rate / 1000).toFixed(1)} kHz)</p>
              <small className="status-meta">
                <span>{m.repo} ·</span>
                {m.cached ? (
                  <span className="inline-badge ok-text"><IconCheckCircle size={13} /> Cached Ready</span>
                ) : (
                  <span className="inline-badge"><IconDownload size={13} /> Not Downloaded</span>
                )}
              </small>
              {!m.cached && (
                <div className="tool-actions">
                  <button
                    className="accent btn-inline"
                    disabled={downloadingModel !== null}
                    onClick={() => void runDownloadModel(m.key)}
                  >
                    <IconDownload size={14} />
                    <span>{downloadingModel === m.key ? "Downloading checkpoint…" : "Download Checkpoint"}</span>
                  </button>
                </div>
              )}
            </article>
          ))}
        </div>
      </div>
    </section>}

    {tab === "compare" && <section className="prepared">
      <label>Shared test script
        <textarea rows={3} value={phrase} disabled={compareBusy} onChange={e => setPhrase(e.target.value)} />
      </label>
      <div className="modal-grid">
        <div>
          <label>Voice A
            <select value={voiceA} disabled={compareBusy} onChange={e => setVoiceA(e.target.value)}>
              {voices.map(v => <optgroup key={v.id} label={v.name}>
                {samples.filter(s => s.voiceId === v.id).map(s =>
                  <option key={s.id} value={s.id}>{v.name} · {s.style}</option>)}
              </optgroup>)}
              <optgroup label="Qwen3 VoiceDesign Presets">
                <option value="qwen:Conversational">Qwen3 · Conversational</option>
                <option value="qwen:Documentary">Qwen3 · Documentary</option>
                <option value="qwen:Energetic">Qwen3 · Energetic</option>
                <option value="qwen:Whisper">Qwen3 · Whisper</option>
              </optgroup>
            </select>
          </label>
          {urlA && <audio controls src={urlA} />}
        </div>
        <div>
          <label>Voice B
            <select value={voiceB} disabled={compareBusy} onChange={e => setVoiceB(e.target.value)}>
              {voices.map(v => <optgroup key={v.id} label={v.name}>
                {samples.filter(s => s.voiceId === v.id).map(s =>
                  <option key={s.id} value={s.id}>{v.name} · {s.style}</option>)}
              </optgroup>)}
              <optgroup label="Qwen3 VoiceDesign Presets">
                <option value="qwen:Conversational">Qwen3 · Conversational</option>
                <option value="qwen:Documentary">Qwen3 · Documentary</option>
                <option value="qwen:Energetic">Qwen3 · Energetic</option>
                <option value="qwen:Whisper">Qwen3 · Whisper</option>
              </optgroup>
            </select>
          </label>
          {urlB && <audio controls src={urlB} />}
        </div>
      </div>
      <div className="tool-actions">
        <button className="accent btn-inline" disabled={!ready || compareBusy || !phrase.trim()} onClick={() => void runCompare()}>
          <IconSwap size={14} />
          <span>{compareBusy ? "Synthesizing A & B…" : "Compare Voices A & B"}</span>
        </button>
      </div>
      {compareError && <p role="alert" className="error">{compareError}</p>}
    </section>}

    {tab === "isolate" && <section className="prepared">
      <p className="hint">Extract clean, leveled speech from noisy recordings using spectral FFT denoising and voice bandpass filtering.</p>
      <div className="modal-grid">
        <label>Upload noisy audio/video
          <input type="file" disabled={isoBusy} accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.webm"
            onChange={e => { setIsoFile(e.target.files?.[0]); setIsoSource(""); }} />
        </label>
        {!!sources.length && <label>Or workspace recording
          <select disabled={isoBusy} value={isoSource} onChange={e => { setIsoSource(e.target.value); setIsoFile(undefined); }}>
            <option value="">Select recording</option>
            {sources.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>}
      </div>
      <div className="tool-actions">
        <button className="accent btn-inline" disabled={isoBusy || (!isoFile && !isoSource)} onClick={() => void runIsolate()}>
          <IconFilter size={14} />
          <span>{isoBusy ? "Isolating vocals…" : "Isolate & Level Vocals"}</span>
        </button>
      </div>
      {isoError && <p role="alert" className="error">{isoError}</p>}
      {isoResult && <div>
        <p className="hint">Isolated vocal track ({isoResult.seconds.toFixed(1)} s)</p>
        <audio controls src={`/media/takes/${isoResult.id}.wav`} />
        <div className="tool-actions">
          <button className="accent" onClick={() => void attachTake(isoResult.id, isoResult.seconds, "[Isolated Vocal Track]", {}).then(close)}>
            Use in focused block
          </button>
        </div>
      </div>}
    </section>}

    {tab === "director" && <section className="prepared">
      <div className="modal-grid">
        <div>
          <h3>Directorial AI Taxonomy Parser</h3>
          <label>Acting direction
            <textarea rows={3} value={directionInput} onChange={e => setDirectionInput(e.target.value)}
              placeholder="e.g. urgent and surprised, intimate whisper" />
          </label>
          <div className="tool-actions">
            <button className="accent btn-inline" disabled={!directionInput.trim()} onClick={() => void runDirection()}>
              <IconDirector size={14} />
              <span>Parse Direction</span>
            </button>
          </div>
          {directionResult && <div className="take-history">
            <article>
              <p><strong>Instruct prompt:</strong> {directionResult.instruct_prompt || "—"}</p>
              <small>
                Rate bias: {directionResult.rate_bias}× · Stability: {directionResult.recommended_stability}% · Style: {directionResult.recommended_style}%
              </small>
              <p className="hint">{directionResult.translate_hint}</p>
              <div className="tool-actions">
                <button className="accent" onClick={() => {
                  applyDirectionToBlock({
                    style: directionResult.instruct_prompt,
                    speed: directionResult.rate_bias,
                    stability: directionResult.recommended_stability,
                    style_exaggeration: directionResult.recommended_style,
                  });
                  close();
                }}>Apply to focused block</button>
              </div>
            </article>
          </div>}
        </div>
        <div>
          <h3>Dubbing Time-Slot Rate Fit</h3>
          <label>Script / translated line
            <textarea rows={3} value={slotText} onChange={e => setSlotText(e.target.value)} />
          </label>
          <label>Target slot duration (seconds)
            <input type="number" min={0.2} max={600} step={0.1} value={slotSeconds}
              onChange={e => setSlotSeconds(e.target.valueAsNumber || 1)} />
          </label>
          <div className="tool-actions">
            <button className="accent btn-inline" disabled={!slotText.trim()} onClick={() => void runRateFit()}>
              <IconClock size={14} />
              <span>Check Slot Fit</span>
            </button>
          </div>
          {rateResult && <div className="take-history">
            <article>
              <p><strong>Verdict: {rateResult.verdict.toUpperCase()}</strong> ({rateResult.word_count} words · est. {rateResult.estimated_seconds}s / {rateResult.slot_seconds}s slot)</p>
              <small>{rateResult.suggestion}</small>
              <div className="tool-actions">
                <button className="accent" onClick={() => {
                  applySpeedToBlock(rateResult.recommended_speed);
                  close();
                }}>Set block speed to {rateResult.recommended_speed}×</button>
              </div>
            </article>
          </div>}
        </div>
      </div>
      {toolError && <p role="alert" className="error">{toolError}</p>}
    </section>}

    {tab === "watermark" && <section className="prepared">
      <p className="hint">Verify whether an audio file contains YouClone’s 16-bit spread-spectrum ultrasonic carrier (18.2–19.6 kHz) and RIFF provenance signature.</p>
      <div className="modal-grid">
        <label>Upload audio file to inspect
          <input type="file" disabled={wmBusy} accept="audio/*,.wav,.mp3,.flac,.m4a"
            onChange={e => { setWmFile(e.target.files?.[0]); setWmSource(""); }} />
        </label>
        {!!sources.length && <label>Or check workspace recording
          <select disabled={wmBusy} value={wmSource} onChange={e => { setWmSource(e.target.value); setWmFile(undefined); }}>
            <option value="">Select recording</option>
            {sources.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>}
      </div>
      <div className="tool-actions">
        <button className="accent btn-inline" disabled={wmBusy || (!wmFile && !wmSource)} onClick={() => void runWatermarkCheck()}>
          <IconShield size={14} />
          <span>{wmBusy ? "Scanning spectrum…" : "Verify Audio Provenance"}</span>
        </button>
      </div>
      {wmError && <p role="alert" className="error">{wmError}</p>}
      {wmResult && <div className="take-history">
        <article>
          <p>
            <strong className="btn-inline">
              {wmResult.detected ? (
                <><IconCheckCircle size={15} className="ok-text" /><span>Verified YouClone Synthetic Audio</span></>
              ) : (
                <><IconInfoCircle size={15} /><span>No YouClone Watermark Detected</span></>
              )}
            </strong>
          </p>
          <small>
            Signature: {wmResult.signature} · Confidence: {(wmResult.confidence * 100).toFixed(1)}%
            {wmResult.bits_matched ? ` · Carrier Bands: ${wmResult.bits_matched}` : ""}
            {wmResult.method ? ` · Method: ${wmResult.method}` : ""}
          </small>
          <p className="hint">{wmResult.details}</p>
        </article>
      </div>}
    </section>}
  </Panel>;
}

export function TakeHistory({ takes, close, select, save }: {
  takes: Take[]; close(): void; select(take: Take): void; save(take: Take, format: Format): Promise<void>;
}) {
  const [query, setQuery] = useState("");
  const [modelFilter, setModelFilter] = useState<string>("all");
  const [format, setFormat] = useState<Format>("wav");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const visible = takes
    .filter(take => (modelFilter === "all" || take.settings.model === modelFilter) &&
      take.settings.text.toLowerCase().includes(query.trim().toLowerCase()))
    .reverse();
  return <Panel title="Take history" subtitle="Search, audition, inspect synthesis metadata, or export any rendered take." close={close}>
    <div className="tool-actions">
      <input aria-label="Search takes" placeholder="Search script text" value={query} onChange={event => setQuery(event.target.value)} />
      <select aria-label="Filter by model" value={modelFilter} onChange={event => setModelFilter(event.target.value)}>
        <option value="all">All models</option>
        <option value="orpheus_seedvc">YouClone Studio (Orpheus-3B + Seed-VC 44kHz)</option>
      </select>
      <select aria-label="Download format" value={format} onChange={event => setFormat(event.target.value as Format)}>
        {(["wav", "mp3", "flac", "m4a"] as const).map(item => <option value={item} key={item}>{item.toUpperCase()}</option>)}
      </select>
    </div>
    {error && <p role="alert" className="error">{error}</p>}
    {!visible.length && <p className="hint">No matching takes.</p>}
    <div className="take-history">{visible.map(take => <article key={take.id}>
      <p>{take.settings.text}</p>
      <small>
        {new Date(take.created).toLocaleString()} · {take.seconds.toFixed(1)} s · {take.settings.model.toUpperCase()} · Speed {take.settings.speed.toFixed(2)}× · Seed {take.settings.seed}
        {take.settings.effect_preset ? ` · ${take.settings.effect_preset}` : ""}
      </small>
      <audio controls preload="none" src={`/media/takes/${take.id}.wav`} />
      <div className="tool-actions">
        <button disabled={pending} onClick={() => select(take)}>Use take</button>
        <button className="btn-inline" disabled={pending} onClick={async () => {
          setPending(true); setError("");
          try { await save(take, format); } catch (problem) { setError(describe(problem)); }
          finally { setPending(false); }
        }}><IconDownload size={14} /><span>Download</span></button>
      </div>
    </article>)}</div>
  </Panel>;
}
