import { useEffect, useRef } from "react";
import { X, MapPin, AlertTriangle, Clock } from "lucide-react";

interface DamageRegion {
  id: string;
  name: string;
  coordinates: { lat: number; lng: number };
  damageClass: "newly-damaged" | "pre-existing" | "undamaged";
  confidence: number;
  dateDetected: string;
  bounds: { x: number; y: number; width: number; height: number };
}

interface LocationDetailPopupProps {
  region: DamageRegion;
  position: { x: number; y: number };
  onClose: () => void;
}

const classLabels: Record<string, string> = {
  "newly-damaged": "Newly Damaged",
  "pre-existing": "Pre-existing Damage",
  "undamaged": "Undamaged",
};

const classColors: Record<string, string> = {
  "newly-damaged": "var(--sentinel-damage-new)",
  "pre-existing": "var(--sentinel-damage-existing)",
  "undamaged": "var(--sentinel-undamaged)",
};

export function LocationDetailPopup({ region, position, onClose }: LocationDetailPopupProps) {
  const popupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  // Adjust position to keep popup in viewport
  const left = Math.min(position.x + 12, window.innerWidth - 320);
  const top = Math.min(position.y + 12, window.innerHeight - 220);

  const color = classColors[region.damageClass];
  const label = classLabels[region.damageClass];

  return (
    <div
      ref={popupRef}
      className="fixed z-50 w-72 rounded-sm border shadow-2xl"
      style={{
        left,
        top,
        backgroundColor: 'var(--sentinel-surface)',
        borderColor: 'var(--sentinel-border)',
        fontFamily: 'var(--font-mono)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b"
        style={{ borderColor: 'var(--sentinel-border)' }}
      >
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
          <span className="text-xs tracking-widest" style={{ color: 'var(--sentinel-text-muted)' }}>
            ZONE DETAIL
          </span>
        </div>
        <button
          onClick={onClose}
          className="transition-colors"
          style={{ color: 'var(--sentinel-text-muted)' }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'var(--sentinel-text-primary)'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--sentinel-text-muted)'}
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="px-4 py-4 space-y-4">
        {/* Zone name */}
        <div>
          <div className="text-sm" style={{ color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-sans)' }}>
            {region.name}
          </div>
        </div>

        {/* Classification */}
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color }} />
          <div>
            <div className="text-xs tracking-wider mb-0.5" style={{ color: 'var(--sentinel-text-muted)' }}>
              CLASSIFICATION
            </div>
            <div className="text-sm" style={{ color, fontFamily: 'var(--font-sans)' }}>
              {label}
            </div>
          </div>
        </div>

        {/* Coordinates */}
        <div className="flex items-center gap-3">
          <MapPin className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--sentinel-text-muted)' }} />
          <div>
            <div className="text-xs tracking-wider mb-0.5" style={{ color: 'var(--sentinel-text-muted)' }}>
              COORDINATES
            </div>
            <div className="text-xs" style={{ color: 'var(--sentinel-text-primary)' }}>
              {region.coordinates.lat.toFixed(4)}°N, {region.coordinates.lng.toFixed(4)}°E
            </div>
          </div>
        </div>

        {/* Date */}
        <div className="flex items-center gap-3">
          <Clock className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--sentinel-text-muted)' }} />
          <div>
            <div className="text-xs tracking-wider mb-0.5" style={{ color: 'var(--sentinel-text-muted)' }}>
              DETECTED
            </div>
            <div className="text-xs" style={{ color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-sans)' }}>
              {region.dateDetected}
            </div>
          </div>
        </div>

        {/* Confidence */}
        <div>
          <div className="flex justify-between text-xs mb-2" style={{ color: 'var(--sentinel-text-muted)' }}>
            <span>CONFIDENCE</span>
            <span style={{ color: 'var(--sentinel-text-primary)' }}>{Math.round(region.confidence * 100)}%</span>
          </div>
          <div className="h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--sentinel-border)' }}>
            <div
              className="h-full rounded-full"
              style={{ width: `${region.confidence * 100}%`, backgroundColor: color }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
