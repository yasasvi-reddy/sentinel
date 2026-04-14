import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { MapPin, Calendar, ChevronDown } from "lucide-react";
import { checkHealth } from "../api";

const CITIES = [
  {
    label: "Kharkiv, Ukraine",
    coords: "49.9935,36.2304",
    startDate: "2021-10-01",
    endDate: "2023-08-31",
  },
  {
    label: "Mariupol, Ukraine",
    coords: "47.0951,37.5397",
    startDate: "2021-10-01",
    endDate: "2023-08-31",
  },
] as const;

export function LandingScreen() {
  const navigate = useNavigate();
  const [selectedCity, setSelectedCity] = useState<typeof CITIES[number] | null>(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [selectedInfrastructure, setSelectedInfrastructure] = useState<string[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  const infrastructureTypes = ["HOSPITALS", "SCHOOLS", "WATER SYSTEMS", "POWER GRID"];

  useEffect(() => {
    checkHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  const handleCityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const city = CITIES.find((c) => c.coords === e.target.value) ?? null;
    setSelectedCity(city);
    setStartDate(city?.startDate ?? "");
    setEndDate(city?.endDate ?? "");
  };

  const toggleInfrastructure = (type: string) => {
    setSelectedInfrastructure((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const handleAnalyze = () => {
    if (selectedCity && startDate && endDate) {
      navigate("/processing", {
        state: { location: selectedCity.coords, startDate, endDate, selectedInfrastructure },
      });
    }
  };

  const pillColor = apiOnline === null
    ? "var(--sentinel-text-muted)"
    : apiOnline
    ? "var(--sentinel-undamaged)"
    : "var(--sentinel-damage-new)";

  const pillLabel = apiOnline === null ? "CONNECTING…" : apiOnline ? "LIVE DATA" : "API OFFLINE";

  return (
    <div className="relative min-h-screen overflow-hidden" style={{
      backgroundColor: 'var(--sentinel-bg)',
      fontFamily: 'var(--font-mono)'
    }}>
      {/* Topographic background decoration */}
      <div className="absolute inset-0 opacity-[0.03]">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="topo" x="0" y="0" width="200" height="200" patternUnits="userSpaceOnUse">
              <path d="M20,100 Q60,80 100,100 T180,100" fill="none" stroke="currentColor" strokeWidth="0.5" />
              <path d="M20,120 Q60,100 100,120 T180,120" fill="none" stroke="currentColor" strokeWidth="0.5" />
              <path d="M20,140 Q60,120 100,140 T180,140" fill="none" stroke="currentColor" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#topo)" style={{ color: 'var(--sentinel-text-primary)' }} />
        </svg>
      </div>

      {/* Scanline overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, var(--sentinel-text-primary) 2px, var(--sentinel-text-primary) 4px)'
        }}
      />

      {/* Live status pill */}
      <div className="absolute top-8 right-8 z-10">
        <div
          className="px-4 py-2 rounded-full border flex items-center gap-2"
          style={{
            backgroundColor: 'var(--sentinel-surface)',
            borderColor: 'var(--sentinel-border)',
            color: 'var(--sentinel-text-primary)'
          }}
        >
          <div className="w-2 h-2 rounded-full relative" style={{ backgroundColor: pillColor }}>
            {apiOnline && (
              <div className="w-2 h-2 rounded-full animate-ping absolute inset-0" style={{ backgroundColor: pillColor }} />
            )}
          </div>
          <span className="text-xs tracking-widest">{pillLabel}</span>
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10 flex items-center justify-center min-h-screen px-8">
        <div className="w-full max-w-2xl">
          {/* Title */}
          <div className="text-center mb-16">
            <h1 className="text-5xl tracking-[0.2em] mb-4" style={{ color: 'var(--sentinel-text-primary)' }}>
              SENTINEL
            </h1>
            <p className="text-sm tracking-[0.15em]" style={{ color: 'var(--sentinel-text-muted)', fontFamily: 'var(--font-sans)' }}>
              War Damage Detection System
            </p>
          </div>

          {/* Input form */}
          <div className="p-8 rounded-sm border" style={{ backgroundColor: 'var(--sentinel-surface)', borderColor: 'var(--sentinel-border)' }}>
            {/* Location dropdown */}
            <div className="mb-6">
              <label className="block text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
                TARGET LOCATION
              </label>
              <div className="relative">
                <MapPin className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none" style={{ color: 'var(--sentinel-text-muted)' }} />
                <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none" style={{ color: 'var(--sentinel-text-muted)' }} />
                <select
                  value={selectedCity?.coords ?? ""}
                  onChange={handleCityChange}
                  className="w-full pl-12 pr-10 py-3 bg-transparent border rounded-sm outline-none transition-colors appearance-none cursor-pointer"
                  style={{
                    borderColor: 'var(--sentinel-border)',
                    color: selectedCity ? 'var(--sentinel-text-primary)' : 'var(--sentinel-text-muted)',
                    fontFamily: 'var(--font-sans)',
                    backgroundColor: 'var(--sentinel-surface)',
                  }}
                  onFocus={(e) => e.target.style.borderColor = 'var(--sentinel-text-muted)'}
                  onBlur={(e) => e.target.style.borderColor = 'var(--sentinel-border)'}
                >
                  <option value="" disabled style={{ color: 'var(--sentinel-text-muted)', backgroundColor: 'var(--sentinel-surface)' }}>
                    Select a city…
                  </option>
                  {CITIES.map((city) => (
                    <option key={city.coords} value={city.coords} style={{ color: 'var(--sentinel-text-primary)', backgroundColor: 'var(--sentinel-surface)' }}>
                      {city.label}
                    </option>
                  ))}
                </select>
              </div>
              {selectedCity && (
                <div className="mt-2 text-xs" style={{ color: 'var(--sentinel-text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {selectedCity.coords}
                </div>
              )}
            </div>

            {/* Date range */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              {[
                { label: "START DATE", value: startDate, set: setStartDate },
                { label: "END DATE",   value: endDate,   set: setEndDate   },
              ].map(({ label, value, set }) => (
                <div key={label}>
                  <label className="block text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
                    {label}
                  </label>
                  <div className="relative">
                    <Calendar className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--sentinel-text-muted)' }} />
                    <input
                      type="date"
                      value={value}
                      onChange={(e) => set(e.target.value)}
                      className="w-full pl-12 pr-4 py-3 bg-transparent border rounded-sm outline-none transition-colors"
                      style={{ borderColor: 'var(--sentinel-border)', color: 'var(--sentinel-text-primary)', fontFamily: 'var(--font-sans)' }}
                      onFocus={(e) => e.target.style.borderColor = 'var(--sentinel-text-muted)'}
                      onBlur={(e) => e.target.style.borderColor = 'var(--sentinel-border)'}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Infrastructure filters */}
            <div className="mb-8">
              <label className="block text-xs tracking-widest mb-3" style={{ color: 'var(--sentinel-text-muted)' }}>
                INFRASTRUCTURE TYPE
              </label>
              <div className="flex flex-wrap gap-2">
                {infrastructureTypes.map((type) => (
                  <button
                    key={type}
                    onClick={() => toggleInfrastructure(type)}
                    className="px-4 py-2 text-xs tracking-wider rounded-full border transition-all"
                    style={{
                      backgroundColor: selectedInfrastructure.includes(type) ? 'var(--sentinel-border)' : 'transparent',
                      borderColor: selectedInfrastructure.includes(type) ? 'var(--sentinel-text-muted)' : 'var(--sentinel-border)',
                      color: selectedInfrastructure.includes(type) ? 'var(--sentinel-text-primary)' : 'var(--sentinel-text-muted)'
                    }}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Analyze button */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedCity || !startDate || !endDate || apiOnline === false}
              className="w-full py-4 text-sm tracking-[0.2em] rounded-sm transition-all disabled:opacity-40"
              style={{ backgroundColor: 'var(--sentinel-text-primary)', color: 'var(--sentinel-bg)' }}
            >
              ANALYZE
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
