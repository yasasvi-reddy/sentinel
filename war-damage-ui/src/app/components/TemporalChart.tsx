import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import type { TemporalPoint } from "../api";

interface TemporalChartProps {
  startDate: string;
  endDate: string;
  data?: TemporalPoint[];
}

export function TemporalChart({ startDate, endDate, data: apiData }: TemporalChartProps) {
  const data = apiData && apiData.length > 0
    ? apiData.map((p) => ({ date: p.date, damage: p.damage_count }))
    : [];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div
          className="p-3 rounded-sm border"
          style={{
            backgroundColor: 'var(--sentinel-surface)',
            borderColor: 'var(--sentinel-border)',
            fontFamily: 'var(--font-sans)'
          }}
        >
          <div className="text-xs mb-1" style={{ color: 'var(--sentinel-text-muted)' }}>
            {payload[0].payload.date}
          </div>
          <div className="text-sm" style={{ color: 'var(--sentinel-text-primary)' }}>
            Damage Index: {Math.round(payload[0].value)}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-32">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="damageGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--sentinel-undamaged)" stopOpacity={0.8} />
              <stop offset="50%" stopColor="var(--sentinel-damage-existing)" stopOpacity={0.8} />
              <stop offset="100%" stopColor="var(--sentinel-damage-new)" stopOpacity={0.8} />
            </linearGradient>
            <linearGradient id="damageGradientFill" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--sentinel-undamaged)" stopOpacity={0.2} />
              <stop offset="50%" stopColor="var(--sentinel-damage-existing)" stopOpacity={0.2} />
              <stop offset="100%" stopColor="var(--sentinel-damage-new)" stopOpacity={0.2} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            stroke="var(--sentinel-text-muted)"
            tick={{
              fill: 'var(--sentinel-text-muted)',
              fontSize: 10,
              fontFamily: 'var(--font-mono)'
            }}
            tickLine={false}
            axisLine={{ stroke: 'var(--sentinel-border)' }}
          />
          <YAxis hide />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="damage"
            stroke="url(#damageGradient)"
            fill="url(#damageGradientFill)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Date range labels */}
      <div
        className="flex justify-between mt-2 text-xs"
        style={{
          color: 'var(--sentinel-text-muted)',
          fontFamily: 'var(--font-mono)'
        }}
      >
        <span>{startDate}</span>
        <span>{endDate}</span>
      </div>
    </div>
  );
}
