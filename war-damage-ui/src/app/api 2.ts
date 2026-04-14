const API_BASE = "http://localhost:8000";

export interface DamageZone {
  polygon: [number, number][];  // [[lat, lng], ...]
  damage_class: 1 | 2;
  label: string;
  confidence: number;
}

export interface TemporalPoint {
  date: string;          // "YYYY-MM"
  damage_count: number;
}

export interface AnalyzeResponse {
  damage_zones: DamageZone[];
  metrics: {
    zones_flagged: number;
    newly_damaged: number;
    pre_existing: number;
    images_analyzed: number;
    total_damage_px: number;
  };
  temporal_progression: TemporalPoint[];
}

export interface HealthResponse {
  status: string;
  gee: boolean;
  unet: boolean;
  vit: boolean;
  device: string;
  timestamp: string;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function analyze(params: {
  location: string;
  start_date: string;
  end_date: string;
  infrastructure_type?: string;
}): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `API error ${res.status}`);
  }
  return res.json();
}
