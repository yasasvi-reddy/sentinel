import { useRef, useEffect, useState, useCallback } from "react";

interface DamageRegion {
  id: string;
  name: string;
  coordinates: { lat: number; lng: number };
  damageClass: "newly-damaged" | "pre-existing" | "undamaged";
  confidence: number;
  dateDetected: string;
  bounds: { x: number; y: number; width: number; height: number };
  polygon?: { x: number; y: number }[];
}

interface DamageMapProps {
  regions: DamageRegion[];
  onRegionClick: (region: DamageRegion, event: React.MouseEvent) => void;
  cityName: string;
  centerCoord?: string;
  postImageB64?: string;  // base64 PNG of post-war satellite composite
  maskB64?: string;       // base64 RGBA PNG of segmentation mask
}

const damageColors = {
  "newly-damaged": "var(--sentinel-damage-new)",
  "pre-existing": "var(--sentinel-damage-existing)",
  "undamaged": "var(--sentinel-undamaged)",
};

export function DamageMap({ regions, onRegionClick, cityName, centerCoord, postImageB64, maskB64 }: DamageMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);
  const [postImg, setPostImg] = useState<HTMLImageElement | null>(null);
  const [maskImg, setMaskImg] = useState<HTMLImageElement | null>(null);

  // Decode satellite image once when the base64 changes — completely separate from drawing
  useEffect(() => {
    if (!postImageB64) { setPostImg(null); return; }
    const img = new Image();
    img.onload = () => setPostImg(img);
    img.onerror = () => setPostImg(null);
    img.src = `data:image/png;base64,${postImageB64}`;
  }, [postImageB64]);

  // Decode mask image once when the base64 changes
  useEffect(() => {
    if (!maskB64) { setMaskImg(null); return; }
    const img = new Image();
    img.onload = () => setMaskImg(img);
    img.onerror = () => setMaskImg(null);
    img.src = `data:image/png;base64,${maskB64}`;
  }, [maskB64]);

  // Draw synchronously — no async, no race conditions
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── Layer 1: satellite background ─────────────────────────────────────
    if (postImg) {
      ctx.globalAlpha = 1.0;
      ctx.drawImage(postImg, 0, 0, canvas.width, canvas.height);
    } else {
      // Fallback: synthetic dark texture
      ctx.fillStyle = "#1a1f2e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const noise = Math.random() * 20;
        data[i] += noise; data[i + 1] += noise; data[i + 2] += noise;
      }
      ctx.putImageData(imageData, 0, 0);
      ctx.strokeStyle = "rgba(45, 52, 72, 0.3)";
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 50) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 50) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }
    }

    // ── Layer 2: segmentation mask at 65% opacity ──────────────────────────
    if (maskImg) {
      ctx.globalAlpha = 0.65;
      ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }

    // ── Layer 3: polygon zone borders ─────────────────────────────────────
    regions.forEach((region) => {
      const color = damageColors[region.damageClass];
      const isHovered = hoveredRegion === region.id;

      ctx.beginPath();
      if (region.polygon && region.polygon.length >= 3) {
        ctx.moveTo(region.polygon[0].x, region.polygon[0].y);
        for (let i = 1; i < region.polygon.length; i++) {
          ctx.lineTo(region.polygon[i].x, region.polygon[i].y);
        }
        ctx.closePath();
      } else {
        const radiusX = region.bounds.width / 2;
        const radiusY = region.bounds.height / 2;
        const centerX = region.bounds.x + radiusX;
        const centerY = region.bounds.y + radiusY;
        for (let i = 0; i <= 20; i++) {
          const angle = (i / 20) * Math.PI * 2;
          const roughness = (Math.random() - 0.5) * 15;
          const x = centerX + Math.cos(angle) * radiusX + roughness;
          const y = centerY + Math.sin(angle) * radiusY + roughness;
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.closePath();
      }
      if (isHovered) {
        ctx.fillStyle = `${color}55`;
        ctx.fill();
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = isHovered ? 2.5 : 1.5;
      ctx.stroke();
    });
  }, [regions, hoveredRegion, postImg, maskImg]);

  useEffect(() => { draw(); }, [draw]);

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const region = regions.find((r) =>
      x >= r.bounds.x && x <= r.bounds.x + r.bounds.width &&
      y >= r.bounds.y && y <= r.bounds.y + r.bounds.height
    );

    setHoveredRegion(region ? region.id : null);
    canvas.style.cursor = region ? "crosshair" : "default";
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const region = regions.find((r) =>
      x >= r.bounds.x && x <= r.bounds.x + r.bounds.width &&
      y >= r.bounds.y && y <= r.bounds.y + r.bounds.height
    );

    if (region) onRegionClick(region, e);
  };

  return (
    <div className="relative w-full h-full" style={{ backgroundColor: 'var(--sentinel-bg)' }}>
      {/* City label */}
      <div
        className="absolute top-6 left-6 z-10 text-sm tracking-wider"
        style={{ color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-mono)' }}
      >
        {cityName.toUpperCase()}
      </div>

      {/* Coordinate / status label */}
      <div
        className="absolute top-6 right-6 z-10 text-xs tracking-wider"
        style={{ color: 'var(--sentinel-text-muted)', fontFamily: 'var(--font-mono)' }}
      >
        {centerCoord || ""}
      </div>

      {/* Satellite source badge */}
      {postImageB64 && (
        <div
          className="absolute bottom-6 right-6 z-10 px-2 py-1 rounded-sm border text-xs tracking-wider"
          style={{
            backgroundColor: 'rgba(13, 15, 20, 0.8)',
            borderColor: 'var(--sentinel-border)',
            color: 'var(--sentinel-text-muted)',
            fontFamily: 'var(--font-mono)'
          }}
        >
          SENTINEL-2 · POST-WAR
        </div>
      )}

      <div
        className="absolute bottom-6 left-6 z-10 text-xs tracking-wider"
        style={{ color: 'var(--sentinel-text-muted)', fontFamily: 'var(--font-mono)' }}
      >
        GRID: UTM 36N
      </div>

      {/* Canvas for map */}
      <canvas
        ref={canvasRef}
        width={1000}
        height={600}
        className="w-full h-full"
        onMouseMove={handleCanvasMouseMove}
        onClick={handleCanvasClick}
      />
    </div>
  );
}
