import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://courtlens.vercel.app"),
  title: "CourtLens | Tennis Video Analysis & Match Stats",
  description: "Upload your tennis match video for ball tracking, point analysis, and a human-reviewed match-stat package from CourtLens.",
  keywords: ["tennis video analysis", "tennis match analysis", "tennis ball tracking", "tennis match statistics", "CourtLens"],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: "/",
    siteName: "CourtLens",
    title: "CourtLens | Tennis Video Analysis & Match Stats",
    description: "Upload your tennis match video for ball tracking and a human-reviewed match-stat package.",
  },
  twitter: {
    card: "summary",
    title: "CourtLens | Tennis Video Analysis & Match Stats",
    description: "Upload your tennis match video for ball tracking and a human-reviewed match-stat package.",
  },
  robots: { index: true, follow: true },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="antialiased">{children}</body>
    </html>
  );
}
