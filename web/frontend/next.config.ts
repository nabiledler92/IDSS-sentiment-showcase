import { withSentryConfig } from "@sentry/nextjs/config";
import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === "development";

// In production, web/vercel.json routes /api/* to the Python service before
// Next.js sees it. In local development, proxy /api/* to the FastAPI server
// started with `uvicorn main:app --port 8000` (see web/README.md).
const API_DEV_ORIGIN = process.env.API_DEV_ORIGIN ?? "http://127.0.0.1:8000";

// Vercel Web Analytics (/_vercel/insights/script.js) and Speed Insights
// (/_vercel/speed-insights/script.js) load from this same origin and post their
// data back to it, so 'self' covers them in production. In development both load
// debug scripts from va.vercel-scripts.com instead.
//
// Google Tag Manager (container GTM-MKG27HR8, shared with nabil-takiyuddin.dev)
// plus GA4 and Microsoft Clarity, which the container loads. Same allow-list as
// the portfolio's next.config.ts; any tag added in GTM later needs its domains
// added here too, or the browser blocks it silently.
const GTM = "https://www.googletagmanager.com";
const GTM_PREVIEW = "https://tagmanager.google.com";
// GA4 sends hits to analytics.google.com and www.google.com (not covered by the
// *.analytics.google.com wildcard), and Google Signals to the visitor's local
// Google domain (google.com.my for most visitors here), per Google's CSP guide.
const GA =
  "https://*.google-analytics.com https://analytics.google.com https://*.analytics.google.com https://*.g.doubleclick.net https://*.google.com https://*.google.com.my";
const CLARITY = "https://*.clarity.ms https://c.bing.com";

const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline' ${GTM} ${GTM_PREVIEW} ${CLARITY}${isDev ? " 'unsafe-eval' https://va.vercel-scripts.com" : ""}`,
  `style-src 'self' 'unsafe-inline' ${GTM_PREVIEW} https://fonts.googleapis.com`,
  `img-src 'self' data: blob: ${GTM} ${GA} ${CLARITY} https://ssl.gstatic.com https://www.gstatic.com`,
  "font-src 'self' https://fonts.gstatic.com",
  `connect-src 'self' ${GTM} ${GA} ${CLARITY}`,
  `frame-src ${GTM}`,
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
  // Serve images as-is. Under Vercel Services the /_next/image optimizer
  // returns 404 for this project, and the only images are two static PNG
  // charts in public/figures, so optimisation adds nothing.
  images: { unoptimized: true },
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

// Sentry (Vercel marketplace integration, shared with nabil-takiyuddin.dev):
// SENTRY_ORG, SENTRY_PROJECT and SENTRY_AUTH_TOKEN are set by the integration
// and used to upload source maps during the build. Browser events go through
// a same-origin tunnel, so the CSP needs no Sentry hosts ('self' covers it).
export default withSentryConfig(nextConfig, {
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  authToken: process.env.SENTRY_AUTH_TOKEN,
  silent: !process.env.CI,
  widenClientFileUpload: true,
  bundleSizeOptimizations: { excludeDebugStatements: true, excludeTracing: true },
});
