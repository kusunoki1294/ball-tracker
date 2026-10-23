import { put } from "@vercel/blob";
import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse } from "next/server";

const MAX_SIZE = 8 * 1024 * 1024 * 1024;
const VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/x-m4v", "application/octet-stream"];

type Submission = {
  submissionId: string;
  name: string;
  email: string;
  matchDate?: string;
  level: string;
  notes?: string;
  fileName: string;
  fileSize: number;
};

function parseSubmission(payload: string | null | undefined): Submission {
  const data = JSON.parse(payload || "{}") as Partial<Submission>;
  const email = String(data.email || "").trim().toLowerCase();
  const submissionId = String(data.submissionId || "");
  if (!/^[0-9a-f-]{36}$/i.test(submissionId) || !String(data.name || "").trim() || !email.includes("@") || !String(data.level || "").trim()) {
    throw new Error("Submission details are incomplete.");
  }
  return {
    submissionId,
    name: String(data.name).trim().slice(0, 120),
    email: email.slice(0, 254),
    matchDate: String(data.matchDate || "").slice(0, 10) || undefined,
    level: String(data.level).trim().slice(0, 40),
    notes: String(data.notes || "").trim().slice(0, 800) || undefined,
    fileName: String(data.fileName || "match-video").slice(0, 180),
    fileSize: Number(data.fileSize || 0),
  };
}

export async function POST(request: Request): Promise<NextResponse> {
  const body = (await request.json()) as HandleUploadBody;
  try {
    const response = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const submission = parseSubmission(clientPayload);
        if (!pathname.startsWith("incoming/") || !pathname.includes(submission.submissionId) || submission.fileSize < 1 || submission.fileSize > MAX_SIZE) {
          throw new Error("Invalid upload request.");
        }
        return {
          allowedContentTypes: VIDEO_TYPES,
          maximumSizeInBytes: MAX_SIZE,
          addRandomSuffix: true,
          validUntil: Date.now() + 10 * 60 * 1000,
          tokenPayload: JSON.stringify(submission),
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        const submission = parseSubmission(tokenPayload);
        await put(`submissions/${submission.submissionId}.json`, JSON.stringify({ ...submission, status: "received", receivedAt: new Date().toISOString(), videoPathname: blob.pathname, videoUrl: blob.url }, null, 2), {
          access: "private",
          contentType: "application/json",
          addRandomSuffix: false,
        });
      },
    });
    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Upload failed." }, { status: 400 });
  }
}
