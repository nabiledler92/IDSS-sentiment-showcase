"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { track } from "@/lib/track";

const TABS = [
  { href: "/", label: "Batch Analyze", key: "batch" },
  { href: "/review", label: "Analyze a Review", key: "review" },
  { href: "/model", label: "Model Info", key: "model" },
];

export function Tabs() {
  const pathname = usePathname();
  return (
    <nav className="tabs" aria-label="Sections">
      {TABS.map((t) => (
        <Link
          key={t.href}
          href={t.href}
          aria-current={pathname === t.href ? "page" : undefined}
          onClick={() => track("tab_view", { tab: t.key })}
        >
          {t.label}
        </Link>
      ))}
    </nav>
  );
}
