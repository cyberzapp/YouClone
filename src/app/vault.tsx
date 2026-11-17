"use client";

import { useEffect, useRef, useState } from "react";
import { describe, type Sample, type Voice } from "./types";

const prompt = "The quick brown fox jumps over the lazy dog. I thought we’d take the early train, but the weather changed our plans. Can you hear the soft rain outside? Let’s make some tea and talk about what comes next.";
const styles = ["Conversational", "Energetic Short-Form", "Calm Documentary", "Whisper"];

export function Vault({ sources, voices, ready, close, saved }: {
  sources: string[]; voices: Voice[]; ready: boolean; close: () => void; saved: (voice: Voice, sample: Sample) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const frame = useRef(0);
  const [source, setSource] = useState(sources[0] ?? "");
  const [file, setFile] = useState<File>();
  const [fileUrl, setFileUrl] = useState("");
  const [name, setName] = useState("");
  const [style, setStyle] = useState(styles[0]);
  const [start, setStart] = useState(sources[0] === "english.mp3" ? 7.6 : 0);
  const [duration, setDuration] = useState(10);
  const [limit, setLimit] = useState(0);
  const [prepared, setPrepared] = useState<Pick<Sample, "id" | "transcript" | "language" | "seconds" | "embedding">>();
  const [cacheEmbedding, setCacheEmbedding] = useState(false);
  const [embedding, setEmbedding] = useState<number[]>();
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [device, setDevice] = useState("");
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(-60);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function stopCapture() {
    cancelAnimationFrame(frame.current);
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null;
    void audioContext.current?.close();
    audioContext.current = null;
  }

  useEffect(() => {
    dialog.current?.showModal();
    navigator.mediaDevices?.enumerateDevices().then(list => setDevices(list.filter(d => d.kind === "audioinput"))).catch(() => {});
    return stopCapture;
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function select(nextSource: string, nextFile?: File) {
    setSource(nextSource); setFile(nextFile); setPrepared(undefined); setEmbedding(undefined); setLimit(0); setStart(nextSource === "english.mp3" ? 7.6 : 0);
  }

  async function record() {
    if (recording) {
      recorder.current?.stop();
      setRecording(false);
      return;
    }
    try {
      setError("");
      const input = await navigator.mediaDevices.getUserMedia({
        audio: { deviceId: device ? { exact: device } : undefined, echoCancellation: false,
          noiseSuppression: false, autoGainControl: false },
      });
      stream.current = input;
      const capture = new MediaRecorder(input);
      recorder.current = capture;
      const chunks: Blob[] = [];
      capture.ondataavailable = event => chunks.push(event.data);
      capture.onstop = () => {
        select("", new File(chunks, "Microphone recording.webm", { type: capture.mimeType }));
        stopCapture();
      };
      const context = new AudioContext();
      audioContext.current = context;
      const analyser = context.createAnalyser();
      context.createMediaStreamSource(input).connect(analyser);
      const buffer = new Float32Array(analyser.fftSize);
      function readLevel() {
        analyser.getFloatTimeDomainData(buffer);
        const rms = Math.sqrt(buffer.reduce((sum, value) => sum + value ** 2, 0) / buffer.length);
        setLevel(Math.max(-60, 20 * Math.log10(rms)));
        frame.current = requestAnimationFrame(readLevel);
      }
      capture.start(250);
      setRecording(true);
      frame.current = requestAnimationFrame(readLevel);
      void navigator.mediaDevices.enumerateDevices().then(list => setDevices(list.filter(d => d.kind === "audioinput"))).catch(() => {});
    } catch (e) {
      stopCapture();
      setError(describe(e));
    }
  }

  async function prepare() {
    setBusy(true); setError(""); setPrepared(undefined);
    try {
      const body = new FormData();
      if (file) body.set("file", file); else body.set("source", source);
      body.set("start", String(start));
      body.set("duration", String(duration));
      body.set("cache_embedding", String(cacheEmbedding));
      const response = await fetch("/api/sample", { method: "POST", body });
      const value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Invalid clip settings.");
      setPrepared(value);
      setEmbedding(undefined);
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!prepared || !name.trim()) return;
    setBusy(true); setError("");
    try {
      const { voice, sample } = await window.studio.sample(name.trim(), {
        ...prepared, style, original: file?.name ?? source, embedding: embedding ?? prepared.embedding,
      });
      saved(voice, sample);
    } catch (e) {
      setError(describe(e));
      setBusy(false);
    }
  }

  const preview = file ? fileUrl : source ? `/media/sources/${encodeURIComponent(source)}` : "";
  return <dialog ref={dialog} onCancel={event => { event.preventDefault(); close(); }}>
    <div className="modal-heading"><div><h2>Voice vault</h2><p>Prepare a recording and review its transcript.</p></div>
      <button onClick={close} aria-label="Close voice vault">Close</button></div>
    <div className="modal-grid">
      <section>
        <label>Workspace recording<select disabled={busy || recording} value={source} onChange={event => select(event.target.value)}>
          <option value="">Select a recording</option>{sources.map(item => <option key={item}>{item}</option>)}
        </select></label>
        <label>Upload audio<input type="file" disabled={busy || recording} accept="audio/*,.wav,.mp3,.m4a,.flac,.webm" onChange={event => {
          if (event.target.files?.[0]) select("", event.target.files[0]);
        }} /></label>
        {preview && <audio key={preview} controls src={preview} onLoadedMetadata={event => {
          const value = event.currentTarget.duration;
          setLimit(Number.isFinite(value) ? value : 0);
        }} />}
        <div className="clip-range">
          <label>Start (seconds)<input type="number" min={0} max={limit || undefined} step=".1" value={start} onChange={event => { setStart(event.target.valueAsNumber || 0); setPrepared(undefined); }} /></label>
          <label>Length (seconds)<input type="number" min={1} max={60} step=".1" value={duration} onChange={event => { setDuration(event.target.valueAsNumber || 0); setPrepared(undefined); }} /></label>
        </div>
        <label className="check"><input type="checkbox" checked={cacheEmbedding} disabled={busy}
          onChange={event => { setCacheEmbedding(event.target.checked); setPrepared(undefined); }} /> Cache ZONOS3 embedding</label>
        <button disabled={!ready || !preview || recording || busy || start < 0 || duration < 1 || duration > 60 || (!!limit && start >= limit)}
          onClick={() => void prepare()}>{busy && !prepared ? "Preparing…" : "Clean and transcribe"}</button>
      </section>
      <section>
        <label>Microphone<select value={device} onChange={event => setDevice(event.target.value)} disabled={recording}>
          <option value="">System default</option>
          {devices.filter(item => item.deviceId && item.deviceId !== "default").map((item, index) =>
            <option value={item.deviceId} key={item.deviceId}>{item.label || `Input ${index + 1}`}</option>)}
        </select></label>
        <label className="meter-label">Input level <output>{recording ? level.toFixed(1) : "—"} dBFS</output>
          <meter min={-60} max={0} low={-12} high={-6} optimum={-9} value={recording ? level : -60} /></label>
        <p className="hint">Aim for −12 to −6 dBFS.</p>
        <p className="script-prompt" lang="en">{prompt}</p>
        <button disabled={busy} onClick={() => void record()}>{recording ? "Stop recording" : "Record sample"}</button>
      </section>
    </div>
    {prepared && <div className="prepared">
      <audio controls src={`/media/samples/${prepared.id}.wav`} />
      <label>Reference transcript<textarea value={prepared.transcript} rows={3}
        onChange={event => setPrepared({ ...prepared, transcript: event.target.value })} /></label>
      <div className="clip-range">
        <label>Profile name<input list="available-voices" value={name} onChange={event => setName(event.target.value)} />
          <datalist id="available-voices">{voices.map(voice => <option key={voice.id}>{voice.name}</option>)}</datalist></label>
        <label>Delivery style<select value={style} onChange={event => setStyle(event.target.value)}>
          {styles.map(item => <option key={item}>{item}</option>)}</select></label>
      </div>
      <label>Import ZONOS3 embedding (optional JSON)
        <input type="file" accept=".json,application/json" onChange={async event => {
          try {
            setEmbedding(undefined);
            const input = event.target.files?.[0];
            if (!input) return;
            if (input.size > 32000) throw new Error("Embedding file is too large.");
            const value: unknown = JSON.parse(await input.text());
            if (!Array.isArray(value) || value.length !== 192 || !value.every(Number.isFinite)) {
              throw new Error("The embedding must be a JSON array of 192 finite numbers.");
            }
            setEmbedding(value); setError("");
          } catch (e) { setError(describe(e)); }
        }} /></label>
      <button className="accent" disabled={busy || !name.trim() || !prepared.transcript.trim()} onClick={() => void save()}>Save sample</button>
    </div>}
    {error && <p className="error" role="alert">{error}</p>}
  </dialog>;
}
