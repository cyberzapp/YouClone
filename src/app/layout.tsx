import type { Metadata } from "next";
import "./studio.css";

export const metadata: Metadata = { title: "YouClone", description: "Local voice recording and speech synthesis studio" };

export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
