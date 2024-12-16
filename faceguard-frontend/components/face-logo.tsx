import Image from "next/image";
import { cn } from "@/lib/utils";

interface FaceLogoProps {
  className?: string;
  width?: number;
  height?: number;
}

export function FaceLogo({ 
  className,
  width = 120,
  height = 120 
}: FaceLogoProps) {
  return (
    <div className={cn("relative", className)}>
      <Image
        src="/logoface-removebg-preview.png"
        alt="FaceGuard Logo"
        width={width}
        height={height}
        className="object-contain"
        priority
      />
    </div>
  );
}
