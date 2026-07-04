"use client";
import { useEffect, useRef } from "react";

export function StarfieldBg() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const count = 180;
    const frag = document.createDocumentFragment();
    for (let i = 0; i < count; i++) {
      const s = document.createElement("div");
      const size = Math.random() * 2 + 0.5;
      s.className = "star";
      s.style.cssText = `
        width:${size}px; height:${size}px;
        left:${Math.random() * 100}%;
        top:${Math.random() * 100}%;
        --d:${3 + Math.random() * 5}s;
        --delay:-${Math.random() * 8}s;
        --min-op:${0.05 + Math.random() * 0.1};
        --max-op:${0.4 + Math.random() * 0.5};
      `;
      frag.appendChild(s);
    }
    el.appendChild(frag);
    return () => { el.innerHTML = ""; };
  }, []);

  return <div ref={ref} className="starfield" aria-hidden="true" />;
}
