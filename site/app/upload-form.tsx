import { FileVideo, UploadCloud } from "lucide-react";

export default function UploadForm() {
  return (
    <div className="upload-form demo-form" aria-label="Demo upload form">
      <div className="demo-banner"><strong>PRODUCT DEMO</strong><span>Uploads are not currently being accepted.</span></div>
      <div className="field-row"><label>YOUR NAME<input disabled placeholder="Alex Morgan" /></label><label>EMAIL FOR RESULTS<input disabled placeholder="alex@example.com" /></label></div>
      <div className="field-row"><label>MATCH DATE<input disabled type="date" /></label><label>PLAYING LEVEL<select disabled defaultValue=""><option value="">Select level</option></select></label></div>
      <label>NOTES <span>(OPTIONAL)</span><textarea disabled rows={3} placeholder="Anything we should know about the match or camera setup?" /></label>
      <div className="drop-zone demo-drop" aria-disabled="true"><FileVideo size={28} /><span><strong>Match video upload</strong><small>MP4, MOV or M4V</small></span></div>
      <button className="submit-button" disabled type="button">Demo only — uploads unavailable <UploadCloud size={19} /></button>
      <p className="form-foot">This interface is a concept preview and does not submit or store information.</p>
    </div>
  );
}
