import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: ["/api/"] },
    sitemap: "https://courtlens.vercel.app/sitemap.xml",
    host: "https://courtlens.vercel.app",
  };
}
