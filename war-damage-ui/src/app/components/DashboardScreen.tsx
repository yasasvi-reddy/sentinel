import { useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { ArrowLeft } from "lucide-react";
import { DamageMap } from "./DamageMap";
import { TemporalChart } from "./TemporalChart";
import { LocationDetailPopup } from "./LocationDetailPopup";
import type { AnalyzeResponse } from "../api";

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

const CANVAS_W = 1000;
const CANVAS_H = 600;
const CANVAS_PAD = 60;

function projectToCanvas(
  lat: number, lng: number,
  minLat: number, maxLat: number, minLng: number, maxLng: number
): { x: number; y: number } {
  const rangeX = maxLng - minLng || 0.001;
  const rangeY = maxLat - minLat || 0.001;
  const x = CANVAS_PAD + ((lng - minLng) / rangeX) * (CANVAS_W - 2 * CANVAS_PAD);
  const y = CANVAS_PAD + ((maxLat - lat) / rangeY) * (CANVAS_H - 2 * CANVAS_PAD);
  return { x, y };
}

function convertZones(apiResult: AnalyzeResponse, endDate: string): DamageRegion[] {
  const zones = apiResult.damage_zones;
  if (zones.length === 0) return [];

  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  for (const z of zones) {
    for (const [lat, lng] of z.polygon) {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
    }
  }

  return zones.map((z, i) => {
    const projected = z.polygon.map(([lat, lng]) =>
      projectToCanvas(lat, lng, minLat, maxLat, minLng, maxLng)
    );
    const xs = projected.map((p) => p.x);
    const ys = projected.map((p) => p.y);
    const bx = Math.min(...xs);
    const by = Math.min(...ys);
    const bw = Math.max(Math.max(...xs) - bx, 10);
    const bh = Math.max(Math.max(...ys) - by, 10);
    const centerLat = z.polygon.reduce((s, [lat]) => s + lat, 0) / z.polygon.length;
    const centerLng = z.polygon.reduce((s, [, lng]) => s + lng, 0) / z.polygon.length;

    return {
      id: String(i),
      name: z.label,
      coordinates: { lat: centerLat, lng: centerLng },
      damageClass: z.damage_class === 1 ? "newly-damaged" : "pre-existing",
      confidence: z.confidence,
      dateDetected: endDate,
      bounds: { x: bx, y: by, width: bw, height: bh },
      polygon: projected,
    } as DamageRegion;
  });
}

export function DashboardScreen() {
  const location = useLocation();
  const navigate = useNavigate();
  const { loc, startDate, endDate, apiResult } = location.state || {} as {
    loc: string; startDate: string; endDate: string; apiResult: AnalyzeResponse;
  };

  const allRegions: DamageRegion[] = apiResult ? convertZones(apiResult, endDate) : [];

  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [activeFilters, setActiveFilters] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<DamageRegion | null>(null);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });

  const infrastructureTypes = ["HOSPITALS", "SCHOOLS", "WATER SYSTEMS", "POWER GRID"];

  const toggleFilter = (type: string) => {
    setActiveFilters((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const filteredRegions = allRegions.filter((r) => r.confidence >= confidenceThreshold);

  const metrics = apiResult
    ? {
        zonesFlagged: apiResult.metrics.zones_flagged,
        newlyDamaged: apiResult.metrics.newly_damaged,
        preExisting: apiResult.metrics.pre_existing,
        imagesAnalyzed: apiResult.metrics.images_analyzed,
      }
    : { zonesFlagged: 0, newlyDamaged: 0, preExisting: 0, imagesAnalyzed: 0 };

  const handleRegionClick = (region: DamageRegion, event: React.MouseEvent) => {
    setSelectedRegion(region);
    setPopupPosition({ x: event.clientX, y: event.clientY });
  };

  return (
    <div
      className="min-h-screen"
      style={{
        backgroundColor: 'var(--sentinel-bg)',
        fontFamily: 'var(--font-mono)'
      }}
    >
      {/* Scanline overlay */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.03] z-0"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, var(--sentinel-text-primary) 2px, var(--sentinel-text-primary) 4px)'
        }}
      />

      <div className="relative z-10 flex h-screen">
        {/* Left sidebar */}
        <div
          className="w-80 border-r p-6 overflow-y-auto"
          style={{
            backgroundColor: 'var(--sentinel-surface)',
            borderColor: 'var(--sentinel-border)'
          }}
        >
          {/* Back button */}
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 mb-8 text-xs tracking-wider transition-colors"
            style={{ color: 'var(--sentinel-text-muted)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--sentinel-text-primary)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--sentinel-text-muted)'}
          >
            <ArrowLeft className="w-4 h-4" />
            NEW ANALYSIS
          </button>

          {/* Location info */}
          <div className="mb-8">
            <div className="text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
              TARGET LOCATION
            </div>
            <div
              className="text-sm mb-4"
              style={{ color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-sans)' }}
            >
              {loc || "—"}
            </div>

            <div className="text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
              DATE RANGE
            </div>
            <div
              className="text-sm"
              style={{ color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-sans)' }}
            >
              {startDate || "—"} — {endDate || "—"}
            </div>
          </div>

          {/* Confidence threshold */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs tracking-widest" style={{ color: 'var(--sentinel-text-muted)' }}>
                CONFIDENCE THRESHOLD
              </span>
              <span className="text-xs" style={{ color: 'var(--sentinel-text-primary)' }}>
                {Math.round(confidenceThreshold * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-full h-1 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, var(--sentinel-undamaged) 0%, var(--sentinel-undamaged) ${confidenceThreshold * 100}%, var(--sentinel-border) ${confidenceThreshold * 100}%, var(--sentinel-border) 100%)`,
              }}
            />
          </div>

          {/* Infrastructure filters */}
          <div className="mb-8">
            <div className="text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
              INFRASTRUCTURE FILTER
            </div>
            <div className="flex flex-wrap gap-2">
              {infrastructureTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleFilter(type)}
                  className="px-3 py-1.5 text-xs tracking-wider rounded-full border transition-all"
                  style={{
                    backgroundColor: activeFilters.includes(type) ? 'var(--sentinel-border)' : 'transparent',
                    borderColor: activeFilters.includes(type) ? 'var(--sentinel-text-muted)' : 'var(--sentinel-border)',
                    color: activeFilters.includes(type) ? 'var(--sentinel-text-primary)' : 'var(--sentinel-text-muted)'
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div>
            <div className="text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
              DAMAGE CLASSIFICATION
            </div>
            <div className="space-y-2">
              {[
                { label: "Newly damaged", color: 'var(--sentinel-damage-new)' },
                { label: "Pre-existing", color: 'var(--sentinel-damage-existing)' },
                { label: "Undamaged", color: 'var(--sentinel-undamaged)' },
              ].map(({ label, color }) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: color }} />
                  <span className="text-xs" style={{ color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-sans)' }}>
                    {label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Map section */}
          <div className="flex-1 relative">
            <DamageMap
              regions={filteredRegions}
              onRegionClick={handleRegionClick}
              cityName={loc || "—"}
              centerCoord={loc || ""}
              postImageB64={apiResult?.post_image_b64}
              maskB64={apiResult?.mask_b64}
            />

            {/* Metrics overlay */}
            <div className="absolute bottom-6 left-6 right-6">
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { label: "ZONES FLAGGED", value: metrics.zonesFlagged },
                  { label: "NEWLY DAMAGED", value: metrics.newlyDamaged },
                  { label: "PRE-EXISTING", value: metrics.preExisting },
                  { label: "IMAGES ANALYZED", value: metrics.imagesAnalyzed },
                ].map((metric, index) => (
                  <div
                    key={index}
                    className="p-4 rounded-sm border backdrop-blur-sm"
                    style={{
                      backgroundColor: 'rgba(22, 27, 39, 0.9)',
                      borderColor: 'var(--sentinel-border)',
                    }}
                  >
                    <div
                      className="text-xs tracking-widest mb-2"
                      style={{ color: 'var(--sentinel-text-muted)' }}
                    >
                      {metric.label}
                    </div>
                    <div className="text-2xl" style={{ color: 'var(--sentinel-text-primary)' }}>
                      {metric.value}
                    </div>
                  </div>
                ))}
              </div>

              {/* Temporal chart */}
              <div
                className="p-6 rounded-sm border backdrop-blur-sm"
                style={{
                  backgroundColor: 'rgba(22, 27, 39, 0.9)',
                  borderColor: 'var(--sentinel-border)',
                }}
              >
                <div
                  className="text-xs tracking-widest mb-4"
                  style={{ color: 'var(--sentinel-text-muted)' }}
                >
                  TEMPORAL PROGRESSION
                </div>
                <TemporalChart
                  startDate={startDate || ""}
                  endDate={endDate || ""}
                  data={apiResult?.temporal_progression}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Location detail popup */}
      {selectedRegion && (
        <LocationDetailPopup
          region={selectedRegion}
          position={popupPosition}
          onClose={() => setSelectedRegion(null)}
        />
      )}
    </div>
  );
}
