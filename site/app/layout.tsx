import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Baseline — Tennis video analysis",
  description: "Send in your tennis match and get back a clear, human-reviewed analysis package.",
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
