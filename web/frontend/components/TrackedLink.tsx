"use client";

import { track } from "@/lib/track";

export type OutboundLinkType = "case_study" | "source_code" | "streamlit";

/** An external link that records an `outbound_click` event with its type. */
export function TrackedLink({
  href,
  linkType,
  children,
}: {
  href: string;
  linkType: OutboundLinkType;
  children: React.ReactNode;
}) {
  return (
    <a href={href} onClick={() => track("outbound_click", { link_type: linkType })}>
      {children}
    </a>
  );
}
