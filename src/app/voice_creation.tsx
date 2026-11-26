"use client";

import { useEffect, useRef, useState } from "react";
import {
  IconArrowLeft,
  IconCheck,
  IconClose,
  IconPause,
  IconPlay,
  IconTrash,
  IconUpload,
  IconWaveform,
} from "./icons";
import { describe, type PvcTrainingState, type StudioVoice } from "./types";

type UploadedItem = { file: File; seconds: number };

async function probeAudioSeconds(file: File): Promise<number> {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      const d = Number.isFinite(audio.duration) ? audio.duration : 0;
      URL.revokeObjectURL(url);
      resolve(d);
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(0);
    };
    audio.src = url;
  });
}

function formatMinSec(totalSec: number): string {
  const s = Math.max(0, Math.round(totalSec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, "0")}`;
}

export function VoiceCreationModal({
  voices,
  activeVoiceId,
  training,
  ready,
  close,
  onSelectVoice,
  onRefresh,
}: {
  voices: StudioVoice[];
  activeVoiceId: string;
  training?: PvcTrainingState;
  ready: boolean;
  close: () => void;
  onSelectVoice: (voice: StudioVoice) => void;
  onRefresh: () => Promise<void>;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previewAudio = useRef<HTMLAudioElement | null>(null);
  const [view, setView] = useState<"hub" | "instant" | "professional">("hub");
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [gender, setGender] = useState<"auto" | "male" | "female">("auto");
  const [pvcSteps, setPvcSteps] = useState(500);
  const [files, setFiles] = useState<UploadedItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [statusNote, setStatusNote] = useState("");

  useEffect(() => {
    dialogRef.current?.showModal();
    return () => {
      previewAudio.current?.pause();
    };
  }, []);

  // Poll live PVC training progress while modal is open and training is running
  useEffect(() => {
    if (!training?.running) return;
    const id = setInterval(() => {
      void onRefresh();
    }, 2000);
    return () => clearInterval(id);
  }, [training?.running, onRefresh]);

  const totalSec = files.reduce((acc, item) => acc + item.seconds, 0);
  const totalMin = totalSec / 60;

  function togglePreview(voice: StudioVoice | undefined, idKey: string) {
    if (!voice?.preview_file) return;
    if (playingId === idKey) {
      previewAudio.current?.pause();
      setPlayingId(null);
      return;
    }
    previewAudio.current?.pause();
    const audio = new Audio(`/media/voices/${voice.preview_file}`);
    audio.onended = () => setPlayingId(null);
    audio.onerror = () => setPlayingId(null);
    previewAudio.current = audio;
    setPlayingId(idKey);
    void audio.play().catch(() => setPlayingId(null));
  }

  async function addFiles(list: FileList | null) {
    if (!list?.length) return;
    setError("");
    const next: UploadedItem[] = [];
    for (const f of Array.from(list)) {
      const sec = await probeAudioSeconds(f);
      next.push({ file: f, seconds: sec });
    }
    setFiles(prev => [...prev, ...next]);
  }

  async function submitInstant() {
    if (!name.trim() || !files.length) return;
    setBusy(true);
    setError("");
    setStatusNote("Analyzing pitch profile and creating Instant Voice Clone…");
    try {
      const fd = new FormData();
      fd.set("name", name.trim());
      fd.set("gender", gender);
      for (const item of files) fd.append("files", item.file);
      const res = await fetch("/api/studio-voices/instant", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Instant voice creation failed.");
      await onRefresh();
      if (data.voice) onSelectVoice(data.voice);
      setFiles([]);
      setName("");
      setView("hub");
      setStatusNote(`Instant Voice "${data.voice?.name}" created and activated!`);
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function submitProfessional() {
    if (!name.trim() || !files.length) return;
    if (totalSec < 1200) {
      setError(`Professional Voice Clone requires at least 20 minutes of audio (currently ${totalMin.toFixed(1)} min).`);
      return;
    }
    setBusy(true);
    setError("");
    setStatusNote("Uploading recordings and starting 2-stage Professional Voice Clone training…");
    try {
      const fd = new FormData();
      fd.set("name", name.trim());
      fd.set("steps", String(pvcSteps));
      for (const item of files) fd.append("files", item.file);
      const res = await fetch("/api/studio-voices/professional", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Failed to start PVC training.");
      await onRefresh();
      setFiles([]);
      setName("");
      setView("hub");
      setStatusNote("Professional Voice Clone training started in the background.");
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function removeVoice(voiceId: string) {
    setError("");
    try {
      const res = await fetch(`/api/studio-voices/${encodeURIComponent(voiceId)}`, { method: "DELETE" });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || "Could not delete voice.");
      }
      await onRefresh();
    } catch (e) {
      setError(describe(e));
    }
  }

  const proPreviewVoice = voices.find(v => v.mode === "professional") ?? voices[0];
  const instantPreviewVoice = voices.find(v => v.mode === "instant") ?? proPreviewVoice;

  return (
    <dialog
      ref={dialogRef}
      className="vc-modal"
      onCancel={e => {
        e.preventDefault();
        close();
      }}
    >
      <div className="vc-header">
        <div>
          <h2>{view === "hub" ? "Voice Creation" : view === "instant" ? "Instant Voice Clone" : "Professional Voice Clone"}</h2>
          <p>
            {view === "hub"
              ? "Clone any voice locally on Apple Silicon — fast Instant Clone or studio-grade Professional Clone."
              : view === "instant"
                ? "Upload 2–5 minutes of clean speech for rapid voice cloning with expressive pacing."
                : "Upload at least 20 minutes of clean audio to match exact expressions, modulations, articulations, and 98–99% timbre."}
          </p>
        </div>
        <div className="vc-header-right">
          <span className="vc-count-pill">
            <i className="vc-count-dot" />
            {voices.length} {voices.length === 1 ? "voice" : "voices"}
          </span>
          {view !== "hub" && (
            <button
              type="button"
              className="btn-inline"
              disabled={busy}
              onClick={() => { setView("hub"); setError(""); setFiles([]); }}
            >
              <IconArrowLeft size={14} />
              <span>Back</span>
            </button>
          )}
          <button type="button" className="icon-btn" onClick={close} aria-label="Close voice creation">
            <IconClose size={15} />
          </button>
        </div>
      </div>

      {training?.running && (
        <div className="pvc-banner" role="status">
          <div className="pvc-banner-top">
            <strong className="btn-inline">
              <IconWaveform size={15} />
              <span>Training Professional Voice · {training.stage}</span>
            </strong>
            <span>{training.percent}%</span>
          </div>
          <div className="pvc-bar"><div style={{ width: `${Math.max(3, training.percent)}%` }} /></div>
          <small>{training.message}</small>
        </div>
      )}
      {training?.error && !training.running && (
        <p className="error" role="alert">Last PVC training error: {training.error}</p>
      )}

      {view === "hub" && (
        <>
          <div className="vc-cards">
            <article className="vc-card vc-card-instant">
              <div className="vc-card-copy">
                <div className="vc-card-title">
                  <h3>Instant Voice Clone</h3>
                  <span className="vc-badge instant">2–5 MIN</span>
                </div>
                <p>
                  Clone your voice fast with 2 to 5 minutes of clean sample audio. Auto-locks your true pitch range and
                  works immediately with expressive delivery.
                </p>
              </div>
              <div className="vc-orb-stage">
                <svg className="vc-arc" viewBox="0 0 220 130" aria-hidden="true">
                  <path
                    d="M 24 118 A 86 86 0 0 1 196 118"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="2 7"
                    strokeLinecap="round"
                  />
                </svg>
                <button
                  type="button"
                  className="vc-orb orb-instant"
                  title="Preview instant voice"
                  aria-label={playingId === "card-instant" ? "Pause instant voice preview" : "Play instant voice preview"}
                  onClick={() => togglePreview(instantPreviewVoice, "card-instant")}
                >
                  <span className="vc-play-btn">
                    {playingId === "card-instant" ? <IconPause size={14} /> : <IconPlay size={14} />}
                  </span>
                </button>
                <div className="vc-gauge-labels">
                  <div><span>Min</span><strong>2 min</strong></div>
                  <div><span>Best</span><strong>5 min</strong></div>
                </div>
              </div>
              <button
                type="button"
                className="accent vc-card-cta"
                disabled={!ready}
                onClick={() => { setError(""); setStatusNote(""); setFiles([]); setName(""); setView("instant"); }}
              >
                Create Instant Voice
              </button>
            </article>

            <article className="vc-card vc-card-pro">
              <div className="vc-card-copy">
                <div className="vc-card-title">
                  <h3>Professional Voice Clone</h3>
                  <span className="vc-badge pro">20+ MIN · PVC</span>
                </div>
                <p>
                  The most realistic local clone. Learns your exact talking expressions, modulations, and articulations
                  via 2-stage Orpheus QLoRA + 44.1 kHz Seed-VC fine-tuning.
                </p>
              </div>
              <div className="vc-orb-stage">
                <svg className="vc-arc" viewBox="0 0 220 130" aria-hidden="true">
                  <path
                    d="M 24 118 A 86 86 0 0 1 196 118"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="2 7"
                    strokeLinecap="round"
                  />
                </svg>
                <button
                  type="button"
                  className="vc-orb orb-pro"
                  title="Preview professional voice"
                  aria-label={playingId === "card-pro" ? "Pause professional voice preview" : "Play professional voice preview"}
                  onClick={() => togglePreview(proPreviewVoice, "card-pro")}
                >
                  <span className="vc-play-btn">
                    {playingId === "card-pro" ? <IconPause size={14} /> : <IconPlay size={14} />}
                  </span>
                </button>
                <div className="vc-gauge-labels">
                  <div><span>Min</span><strong>20 min</strong></div>
                  <div><span>Best</span><strong>30+ min</strong></div>
                </div>
              </div>
              <button
                type="button"
                className="vc-card-cta pro-cta"
                disabled={!ready || training?.running}
                onClick={() => { setError(""); setStatusNote(""); setFiles([]); setName(""); setView("professional"); }}
              >
                {training?.running ? "Training in Progress…" : "Train Professional Voice"}
              </button>
            </article>
          </div>

          <section className="vc-library">
            <div className="vc-library-head">
              <h3>Your Studio Voices</h3>
              <span>Click any voice to make it active in the composer</span>
            </div>
            <div className="vc-voice-list">
              {voices.map(v => {
                const isActive = v.id === activeVoiceId;
                return (
                  <div key={v.id} className={isActive ? "vc-voice-row active" : "vc-voice-row"}>
                    <button
                      type="button"
                      className={v.mode === "professional" ? "vc-mini-orb orb-pro" : "vc-mini-orb orb-instant"}
                      onClick={() => togglePreview(v, v.id)}
                      title="Play preview"
                      aria-label={playingId === v.id ? `Pause ${v.name} preview` : `Play ${v.name} preview`}
                    >
                      {playingId === v.id ? <IconPause size={12} /> : <IconPlay size={12} />}
                    </button>
                    <div className="vc-voice-meta">
                      <div className="vc-voice-name">
                        <strong>{v.name}</strong>
                        <span className={v.mode === "professional" ? "vc-badge pro" : "vc-badge instant"}>
                          {v.mode === "professional" ? "PRO" : "INSTANT"}
                        </span>
                        {v.target_f0_hz ? <span className="vc-chip">{Math.round(v.target_f0_hz)} Hz</span> : null}
                        {v.sample_minutes ? <span className="vc-chip">{v.sample_minutes.toFixed(1)} min</span> : null}
                      </div>
                      <small>{v.description || "Ready for local synthesis"}</small>
                    </div>
                    <div className="vc-voice-actions">
                      <button
                        type="button"
                        className={isActive ? "accent btn-inline" : "btn-inline"}
                        onClick={() => { onSelectVoice(v); close(); }}
                      >
                        {isActive ? <><IconCheck size={13} /><span>Active</span></> : <span>Use Voice</span>}
                      </button>
                      {!v.is_default && (
                        <button
                          type="button"
                          className="icon-btn danger-hover"
                          title="Delete voice"
                          aria-label={`Delete ${v.name}`}
                          onClick={() => void removeVoice(v.id)}
                        >
                          <IconTrash size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        </>
      )}

      {view === "instant" && (
        <div className="vc-form">
          <div className="vc-form-grid">
            <label>
              Voice name
              <input
                value={name}
                disabled={busy}
                placeholder="e.g. Alex · Creator Voice"
                onChange={e => setName(e.target.value)}
              />
            </label>
            <label>
              Vocal character (Actor matching)
              <select value={gender} disabled={busy} onChange={e => setGender(e.target.value as "auto" | "male" | "female")}>
                <option value="auto">Auto-detect from audio pitch</option>
                <option value="male">Male expressive actor</option>
                <option value="female">Female expressive actor</option>
              </select>
            </label>
          </div>

          <label className="vc-dropzone">
            <IconUpload size={20} className="dropzone-icon" />
            <strong>Upload clean audio recordings (2–5 minutes recommended)</strong>
            <span>WAV, MP3, M4A, FLAC · Single or multiple files · Quiet room, no background music</span>
            <input
              type="file"
              multiple
              disabled={busy}
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={e => void addFiles(e.target.files)}
            />
          </label>

          <div className="vc-meter-card">
            <div className="vc-meter-row">
              <span>Uploaded audio duration</span>
              <strong className={totalSec >= 120 ? "ok-text btn-inline" : "btn-inline"}>
                <span>{formatMinSec(totalSec)} ({totalMin.toFixed(1)} min) ·</span>
                {totalSec >= 120 ? (
                  <><IconCheck size={13} /><span>Ideal 2–5 min range</span></>
                ) : totalSec >= 15 ? (
                  <span>Usable (2+ min recommended)</span>
                ) : (
                  <span>Min 15s required</span>
                )}
              </strong>
            </div>
            <div className="pvc-bar">
              <div style={{ width: `${Math.min(100, (totalSec / 300) * 100)}%` }} />
            </div>
          </div>

          {files.length > 0 && (
            <ul className="vc-file-list">
              {files.map((item, idx) => (
                <li key={`${item.file.name}-${idx}`}>
                  <span>{item.file.name}</span>
                  <span className="vc-file-right">
                    <span>{formatMinSec(item.seconds)}</span>
                    <button
                      type="button"
                      className="icon-btn sm"
                      aria-label={`Remove ${item.file.name}`}
                      disabled={busy}
                      onClick={() => setFiles(prev => prev.filter((_, i) => i !== idx))}
                    >
                      <IconClose size={12} />
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="vc-form-footer">
            <button
              type="button"
              className="accent"
              disabled={busy || !name.trim() || totalSec < 15}
              onClick={() => void submitInstant()}
            >
              {busy ? "Creating Instant Voice…" : "Create Instant Voice"}
            </button>
          </div>
        </div>
      )}

      {view === "professional" && (
        <div className="vc-form">
          <div className="vc-form-grid">
            <label>
              Voice name
              <input
                value={name}
                disabled={busy}
                placeholder="e.g. Alex · Studio PVC"
                onChange={e => setName(e.target.value)}
              />
            </label>
            <label>
              Training depth (local Apple Silicon)
              <select value={pvcSteps} disabled={busy} onChange={e => setPvcSteps(Number(e.target.value))}>
                <option value={500}>Standard PVC · 500 steps (~25 min)</option>
                <option value={800}>High Likeness · 800 steps (~40 min)</option>
                <option value={1000}>Maximum PVC · 1000 steps (~50 min)</option>
              </select>
            </label>
          </div>

          <label className="vc-dropzone pro-dropzone">
            <IconUpload size={20} className="dropzone-icon" />
            <strong>Upload at least 20 minutes of your voice recordings</strong>
            <span>Include expressive talking, storytelling, and natural pauses · WAV / MP3 / M4A / FLAC (up to 1 GB total)</span>
            <input
              type="file"
              multiple
              disabled={busy}
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={e => void addFiles(e.target.files)}
            />
          </label>

          <div className="vc-meter-card">
            <div className="vc-meter-row">
              <span>Professional Voice Clone minimum</span>
              <strong className={totalSec >= 1200 ? "ok-text btn-inline" : "btn-inline"}>
                <span>{totalMin.toFixed(1)} / 20.0 min ·</span>
                {totalSec >= 1200 ? (
                  <><IconCheck size={13} /><span>Ready for 2-Stage PVC Training</span></>
                ) : (
                  <span>Need {(20 - totalMin).toFixed(1)} more min</span>
                )}
              </strong>
            </div>
            <div className="pvc-bar">
              <div style={{ width: `${Math.min(100, (totalSec / 1200) * 100)}%` }} />
            </div>
          </div>

          {files.length > 0 && (
            <ul className="vc-file-list">
              {files.map((item, idx) => (
                <li key={`${item.file.name}-${idx}`}>
                  <span>{item.file.name}</span>
                  <span className="vc-file-right">
                    <span>{formatMinSec(item.seconds)}</span>
                    <button
                      type="button"
                      className="icon-btn sm"
                      aria-label={`Remove ${item.file.name}`}
                      disabled={busy}
                      onClick={() => setFiles(prev => prev.filter((_, i) => i !== idx))}
                    >
                      <IconClose size={12} />
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="vc-form-footer">
            <button
              type="button"
              className="accent"
              disabled={busy || !name.trim() || totalSec < 1200 || training?.running}
              onClick={() => void submitProfessional()}
            >
              {busy ? "Uploading & Starting Training…" : "Start 2-Stage Professional Training"}
            </button>
          </div>
        </div>
      )}

      {statusNote && <p className="notice">{statusNote}</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </dialog>
  );
}
