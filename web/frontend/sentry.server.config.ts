// Sentry on the Next.js Node.js server: server components and the tunnel that
// forwards browser events to Sentry. (The Python API in web/backend isn't
// covered; it logs its own errors to the Vercel runtime logs.)
import * as Sentry from "@sentry/nextjs";
import { sentryDataCollection, sentryInitialScope } from "@/lib/sentry-privacy";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.VERCEL_ENV ?? "development",
  // Answers the browser's `tunnel: true` requests and forwards them to Sentry.
  integrations: [Sentry.serveTunnelIntegration()],
  tracesSampleRate: 0,
  // No personal data: see lib/sentry-privacy.ts.
  dataCollection: sentryDataCollection,
  initialScope: sentryInitialScope,
});
