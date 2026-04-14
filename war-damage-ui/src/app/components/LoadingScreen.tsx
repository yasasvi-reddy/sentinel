import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router";
import { motion } from "motion/react";
import { analyze } from "../api";
import type { AnalyzeResponse } from "../api";

const STEPS = [
  "Fetching Sentinel-2 imagery…",
  "Applying cloud masking…",
  "Running U-Net segmentation…",
  "Classifying temporal changes with ViT…",
  "Vectorising damage zones…",
  "Generating assessment report…",
];

export function LoadingScreen() {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { location: loc, startDate, endDate, selectedInfrastructure } = state || {};

  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [apiResult, setApiResult] = useState<AnalyzeResponse | null>(null);
  const calledRef = useRef(false);

  // Drive navigation from a useEffect so it fires inside React's lifecycle,
  // where React Router v7's data router reliably processes it.
  useEffect(() => {
    if (!apiResult) return;
    console.log("[LoadingScreen] apiResult received, scheduling navigation", apiResult);
    const timer = setTimeout(() => {
      console.log("[LoadingScreen] navigating to /results");
      navigate("/results", { state: { loc, startDate, endDate, apiResult } });
    }, 600);
    return () => clearTimeout(timer);
  }, [apiResult]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (calledRef.current) return;
    calledRef.current = true;

    // Advance the visual step ticker independently of the real API call
    let step = 0;
    const ticker = setInterval(() => {
      if (step < STEPS.length - 1) {
        setCompletedSteps((prev) => [...prev, STEPS[step]]);
        step++;
        setCurrentStep(step);
      }
    }, 2500);

    const infraType = selectedInfrastructure?.length > 0
      ? selectedInfrastructure.join(",").toLowerCase()
      : "all";

    analyze({
      location: loc,
      start_date: startDate,
      end_date: endDate,
      infrastructure_type: infraType,
    })
      .then((result) => {
        console.log("[LoadingScreen] API call succeeded", result);
        clearInterval(ticker);
        // Flush remaining steps instantly, then signal the navigation effect
        setCompletedSteps(STEPS.slice(0, -1));
        setCurrentStep(STEPS.length - 1);
        setApiResult(result);
      })
      .catch((err: Error) => {
        console.error("[LoadingScreen] API call failed", err);
        clearInterval(ticker);
        setError(err.message);
      });

    return () => clearInterval(ticker);
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const progress = error
    ? 0
    : ((currentStep + 1) / STEPS.length) * 100;

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-8"
      style={{ backgroundColor: 'var(--sentinel-bg)', fontFamily: 'var(--font-mono)' }}
    >
      {/* Scanline overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{ backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, var(--sentinel-text-primary) 2px, var(--sentinel-text-primary) 4px)' }}
      />

      <div className="relative z-10 w-full max-w-2xl">
        {/* Satellite orbit animation */}
        <div className="mb-16 flex justify-center">
          <div className="relative w-64 h-64">
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 200" style={{ transform: 'rotate(-20deg)' }}>
              <ellipse cx="100" cy="100" rx="80" ry="40" fill="none"
                stroke="var(--sentinel-border)" strokeWidth="1" strokeDasharray="4 4" />
            </svg>
            <motion.div
              className="absolute w-3 h-3 rounded-full"
              animate={{ offsetDistance: ["0%", "100%"] }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              style={{
                backgroundColor: error ? 'var(--sentinel-damage-new)' : 'var(--sentinel-text-primary)',
                boxShadow: `0 0 12px ${error ? 'var(--sentinel-damage-new)' : 'var(--sentinel-text-primary)'}`,
                offsetPath: "path('M 100,100 m -80,-20 a 80,40 0 1,0 160,0 a 80,40 0 1,0 -160,0')",
                offsetRotate: "0deg",
              }}
            />
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full border-2"
              style={{ borderColor: 'var(--sentinel-border)', backgroundColor: 'var(--sentinel-surface)' }}
            >
              <div className="w-full h-full rounded-full" style={{ background: 'radial-gradient(circle at 30% 30%, var(--sentinel-border), var(--sentinel-surface))' }} />
            </div>
          </div>
        </div>

        {/* Terminal readout */}
        <div className="p-6 rounded-sm border" style={{ backgroundColor: 'var(--sentinel-surface)', borderColor: 'var(--sentinel-border)' }}>
          <div
            className="text-xs tracking-widest mb-4 pb-3 border-b"
            style={{ color: 'var(--sentinel-text-muted)', borderColor: 'var(--sentinel-border)' }}
          >
            PROCESSING STATUS
          </div>

          <div className="space-y-2 min-h-[200px]">
            {completedSteps.map((log, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
                className="flex items-start gap-3"
                style={{ color: 'var(--sentinel-text-primary)' }}
              >
                <span className="text-xs" style={{ color: 'var(--sentinel-undamaged)' }}>✓</span>
                <span className="text-sm" style={{ fontFamily: 'var(--font-sans)' }}>{log}</span>
              </motion.div>
            ))}

            {error ? (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-start gap-3"
              >
                <span className="text-xs" style={{ color: 'var(--sentinel-damage-new)' }}>✕</span>
                <div>
                  <span className="text-sm" style={{ color: 'var(--sentinel-damage-new)', fontFamily: 'var(--font-sans)' }}>
                    {error}
                  </span>
                  <button
                    onClick={() => navigate("/")}
                    className="block mt-3 text-xs tracking-wider underline"
                    style={{ color: 'var(--sentinel-text-muted)' }}
                  >
                    ← Return to search
                  </button>
                </div>
              </motion.div>
            ) : currentStep < STEPS.length && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1, repeat: Infinity }}
                className="flex items-start gap-3"
                style={{ color: 'var(--sentinel-text-primary)' }}
              >
                <span className="text-xs" style={{ color: 'var(--sentinel-undamaged)' }}>&gt;</span>
                <span className="text-sm" style={{ fontFamily: 'var(--font-sans)' }}>{STEPS[currentStep]}</span>
              </motion.div>
            )}
          </div>

          {/* Progress bar */}
          <div className="mt-6 pt-4 border-t" style={{ borderColor: 'var(--sentinel-border)' }}>
            <div className="flex justify-between text-xs mb-2" style={{ color: 'var(--sentinel-text-muted)' }}>
              <span>PROGRESS</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--sentinel-border)' }}>
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: error ? 'var(--sentinel-damage-new)' : 'var(--sentinel-undamaged)' }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
