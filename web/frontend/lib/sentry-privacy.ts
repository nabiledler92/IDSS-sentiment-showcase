import type * as Sentry from "@sentry/nextjs";

/**
 * What Sentry may collect, shared by the browser, server and edge setups.
 *
 * Sentry v11 collects request bodies, headers, cookies, user info and stack
 * variables by default. Here that would include customer reviews typed into
 * Analyze a Review, uploaded review files and AI questions, so all of it is
 * switched off: an error report carries the error, the stack trace and the
 * page URL only. Same settings as nabil-takiyuddin.dev.
 */
export const sentryDataCollection: NonNullable<Parameters<typeof Sentry.init>[0]>["dataCollection"] = {
  userInfo: false,
  cookies: false,
  httpHeaders: false,
  httpBodies: [],
  urlQueryParams: false,
  stackFrameVariables: false,
  genAI: { inputs: false, outputs: false },
};

/**
 * Both sites report to one Sentry project, so every event carries this tag
 * (the portfolio uses site: "portfolio") to filter issues by site.
 */
export const sentryInitialScope = { tags: { site: "idss-web" } };
