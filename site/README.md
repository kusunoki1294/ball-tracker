# Baseline Tennis

Customer-facing upload site for the ball-tracker project. Players submit a match video and contact details; the video and its submission metadata are stored in a private Vercel Blob store for manual processing.

## Local development

```bash
npm install
npm run dev
```

The page can be developed without credentials. Completing an upload requires a connected private Vercel Blob store and the environment values supplied by Vercel.

## Vercel setup

1. Import this repository into Vercel.
2. Set the project Root Directory to `site`.
3. Create a private Blob store from the project Storage tab and connect it to Production and Preview.
4. Deploy.

Uploaded videos are stored under `incoming/<date>/<receipt>/`. A matching JSON submission record is stored under `submissions/<receipt>.json`.
