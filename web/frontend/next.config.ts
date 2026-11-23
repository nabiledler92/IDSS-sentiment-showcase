import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === "development";

// In production, web/vercel.json routes /api/* to the Python service before
// Next.js sees it. In local development, proxy /api/* to the FastAPI server
// started with `uvicorn main:app --port 8000` (see web/README.md).
const API_DEV_ORIGIN = process.env.API_DEV_ORIGIN ?? "http://127.0.0.1:8000";

const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  // The portfolio may embed this app on its case study page.
  "frame-ancestors 'self' https://nabil-takiyuddin.dev",
  "upgrade-insecure-requests",
].join("; ");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return isDev ? [{ source: "/api/:path*", destination: `${API_DEV_ORIGIN}/api/:path*` }] : [];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Content-Security-Policy", value: csp },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
