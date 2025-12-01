"use client";

import dynamic from "next/dynamic";
import { Analytics } from "@vercel/analytics/next";

// Dynamic import to avoid SSR issues with onnxruntime-web
const SAMDemo = dynamic(() => import("./SAMDemo"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto mb-4"></div>
        <p>Loading SAM Demo...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return (
    <>
      <SAMDemo />
      <Analytics />
    </>
  );
}
