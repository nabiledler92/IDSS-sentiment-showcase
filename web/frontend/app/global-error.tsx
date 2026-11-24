"use client";

import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";

/** Last-resort error screen for errors in the root layout. Reports them to Sentry. */
export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  // Inline styles: this screen replaces the root layout, so globals.css may not load.
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          background: "#faf6f0",
          color: "#2e2620",
          fontFamily: "system-ui, sans-serif",
          textAlign: "center",
          padding: "24px",
        }}
      >
        <div>
          <h1 style={{ fontFamily: "Georgia, serif", fontSize: "1.6rem", margin: "0 0 12px" }}>Something went wrong.</h1>
          <p style={{ color: "#5a5147", margin: "0 0 20px" }}>The error has been reported. Please try again.</p>
          {/* A plain link on purpose: this screen replaces the root layout, so a full reload is safest. */}
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
          <a href="/" style={{ color: "#8b5e34", fontWeight: 600 }}>
            Back to Batch Analyze
          </a>
        </div>
      </body>
    </html>
  );
}
