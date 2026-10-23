import { ArrowDown, ChartNoAxesCombined, CircleCheck, ScanSearch } from "lucide-react";
import UploadForm from "./upload-form";

const steps = [
  { n: "01", title: "Send your match", text: "Upload one continuous match video and tell me a little about the recording." },
  { n: "02", title: "I run the analysis", text: "I process the footage through the ball-tracking pipeline, then review the output by hand." },
  { n: "03", title: "Get your results", text: "I send the finished video and match-stat package to the email you provide." },
];

export default function Home() {
  return (
    <main>
      <nav className="nav shell"><a className="brand" href="#top" aria-label="CourtLens home"><span className="ball-mark" />COURTLENS</a><a className="nav-cta" href="#upload">Send a match <ArrowDown size={15} /></a></nav>
      <section className="hero shell" id="top">
        <div className="hero-copy"><p className="eyebrow">TENNIS VIDEO ANALYSIS / HUMAN REVIEWED</p><h1>See the match<br />you <em>actually</em> played.</h1><p className="lede">Upload your footage. I’ll track the ball, process the match and send back a clear analysis package built around your game.</p><div className="hero-actions"><a className="button primary" href="#upload">Upload your video <ArrowDown size={17} /></a><span>One match per submission<br />MP4, MOV or M4V</span></div></div>
        <div className="court-card" aria-label="Stylized tennis court showing ball tracking"><div className="court-lines"><i /><b /><span /></div><div className="track-path"><span className="p1" /><span className="p2" /><span className="p3" /><span className="p4" /></div><div className="court-label"><ScanSearch size={18} /><span>Tracking frame by frame</span></div></div>
      </section>
      <section className="ticker" aria-label="Analysis outputs"><div>SHOT PLACEMENT <i /> BALL PATH <i /> POINT TIMELINE <i /> MATCH VIDEO <i /> HUMAN REVIEW <i /> SHOT PLACEMENT <i /> BALL PATH</div></section>
      <section className="process shell" id="process"><div className="section-intro"><p className="eyebrow">THE HANDOFF</p><h2>Your camera.<br />My pipeline.</h2></div><div className="steps">{steps.map((step) => <article key={step.n}><span>{step.n}</span><h3>{step.title}</h3><p>{step.text}</p></article>)}</div></section>
      <section className="output shell"><div className="output-art"><div className="mini-court" /><div className="stat-card"><small>DELIVERABLE</small><strong>Match analysis</strong><p>Tracked video + stats summary</p></div></div><div className="output-copy"><p className="eyebrow">WHAT COMES BACK</p><h2>Not a black box.<br />A reviewed result.</h2><p>The tracking code does the heavy lifting; I check the output before returning it. You receive a processed match video plus the stats the footage supports.</p><ul><li><CircleCheck size={17} /> Ball-tracked match video</li><li><CircleCheck size={17} /> Point and event timeline</li><li><CircleCheck size={17} /> Match-stat summary</li></ul></div></section>
      <section className="upload-section" id="upload"><div className="shell upload-grid"><div className="upload-copy"><p className="eyebrow">SEND YOUR MATCH</p><h2>Ready when<br />you are.</h2><p>A stable, full-court view gives the tracker its best shot. Keep the whole court in frame and avoid moving the camera during play.</p><div className="privacy-note"><ChartNoAxesCombined size={20} /><span><strong>Your footage stays private.</strong><br />It is stored for analysis and isn’t published.</span></div></div><UploadForm /></div></section>
      <footer className="shell"><a className="brand" href="#top"><span className="ball-mark" />COURTLENS</a><p>Tennis analysis, one match at a time.</p><span>© {new Date().getFullYear()}</span></footer>
    </main>
  );
}
