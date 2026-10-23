"use client";

import { ChangeEvent, FormEvent, useState } from "react";
import { Check, FileVideo, LoaderCircle, UploadCloud, X } from "lucide-react";
import { upload } from "@vercel/blob/client";

const MAX_SIZE = 8 * 1024 * 1024 * 1024;

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [message, setMessage] = useState("");
  const [receipt, setReceipt] = useState("");

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.files?.[0] ?? null; setMessage("");
    if (next && next.size > MAX_SIZE) { setFile(null); setMessage("That file is over the 8 GB upload limit."); return; }
    setFile(next);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) { setMessage("Choose a match video first."); return; }
    const form = new FormData(event.currentTarget);
    if (String(form.get("website") || "")) return;
    setPhase("uploading"); setProgress(0); setMessage("");
    try {
      const submissionId = crypto.randomUUID();
      const details = Object.fromEntries(form.entries()); delete details.video; delete details.consent; delete details.website;
      const safeName = file.name.replace(/[^a-zA-Z0-9._-]+/g, "-");
      const pathname = `incoming/${new Date().toISOString().slice(0, 10)}/${submissionId}/${safeName}`;
      await upload(pathname, file, {
        access: "private",
        handleUploadUrl: "/api/uploads",
        multipart: true,
        clientPayload: JSON.stringify({ submissionId, fileName: file.name, fileSize: file.size, ...details }),
        onUploadProgress: ({ percentage }) => setProgress(Math.round(percentage)),
      });
      setProgress(100); setReceipt(submissionId); setPhase("done");
    } catch (error) { setPhase("error"); setMessage(error instanceof Error ? error.message : "Upload failed. Please try again."); }
  }

  if (phase === "done") return <div className="success-card" role="status"><span className="success-icon"><Check size={28} /></span><p className="eyebrow">UPLOAD RECEIVED</p><h3>Your match is in the queue.</h3><p>I’ll use the email you provided to return the finished analysis. Keep this receipt for reference:</p><code>{receipt}</code><button type="button" onClick={() => { setPhase("idle"); setFile(null); setProgress(0); }}>Send another match</button></div>;

  return <form className="upload-form" onSubmit={submit}>
    <div className="field-row"><label>YOUR NAME<input name="name" required autoComplete="name" placeholder="Alex Morgan" /></label><label>EMAIL FOR RESULTS<input name="email" type="email" required autoComplete="email" placeholder="alex@example.com" /></label></div>
    <div className="field-row"><label>MATCH DATE<input name="matchDate" type="date" /></label><label>PLAYING LEVEL<select name="level" required defaultValue=""><option value="" disabled>Select level</option><option>Beginner</option><option>Intermediate</option><option>Advanced</option><option>Competitive</option></select></label></div>
    <label>NOTES <span>(OPTIONAL)</span><textarea name="notes" rows={3} maxLength={800} placeholder="Anything I should know about the match or camera setup?" /></label>
    <label className={`drop-zone ${file ? "has-file" : ""}`}><input name="video" type="file" required accept="video/mp4,video/quicktime,video/x-m4v,.mp4,.mov,.m4v" onChange={chooseFile} />{file ? <><FileVideo size={28} /><span><strong>{file.name}</strong><small>{(file.size / (1024 ** 3)).toFixed(2)} GB</small></span><X size={18} /></> : <><UploadCloud size={29} /><span><strong>Drop your match video here</strong><small>or click to browse · MP4, MOV, M4V · up to 8 GB</small></span></>}</label>
    <label className="consent"><input type="checkbox" name="consent" required /> <span>I have permission to share this footage for private analysis.</span></label><input className="honeypot" name="website" tabIndex={-1} autoComplete="off" />
    {message && <p className="form-error" role="alert">{message}</p>}{phase === "uploading" && <div className="progress-wrap"><div><span>Uploading securely…</span><span>{progress}%</span></div><progress value={progress} max="100" /></div>}
    <button className="submit-button" disabled={phase === "uploading"} type="submit">{phase === "uploading" ? <><LoaderCircle className="spin" size={19} /> Uploading — keep this tab open</> : <>Send match for analysis <UploadCloud size={19} /></>}</button><p className="form-foot">Upload time depends on your file size and connection. Keep this page open until you see a receipt.</p>
  </form>;
}
