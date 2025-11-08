import { useState, useMemo, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import api from "@/api/api";
import { toast } from "sonner";
import Loader from "@/components/Loader";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
// map imports
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
// leaflet.heat plugin augments L with heatLayer
import "leaflet.heat";

const PredictAndReport = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [summary, setSummary] = useState<{ leak_count: number; no_leak_count: number } | null>(null);
  const [heatPoints, setHeatPoints] = useState<Array<[number, number, number, string, number]>>([]);
  const [heatLoading, setHeatLoading] = useState(false);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
  };

  const generateHeatmap = async () => {
    if (!file) return toast.error("Upload the CSV first to generate heatmap");
    setHeatLoading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await api.post('/heatmap_by_zone', form, { headers: { "Content-Type": "multipart/form-data" } });
      if (resp.data && Array.isArray(resp.data.points) && resp.data.points.length > 0) {
        // points: [[lat, lon, weight, zone, max_flow], ...]
        setHeatPoints(resp.data.points);
        toast.success('Heatmap points loaded');
      } else if (resp.data && resp.data.zones) {
        toast.info('No spatial coords found in CSV; received zone stats. Provide lat/lon for heatmap.');
        setHeatPoints([]);
      } else {
        toast.error('No heatmap points returned');
      }
    } catch (err: any) {
      console.error(err);
      toast.error(err.response?.data?.error || 'Heatmap generation failed');
    } finally {
      setHeatLoading(false);
    }
  };

  const runPredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return toast.error("Select a CSV file to predict");
    setLoading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await api.post("/predict", form, { headers: { "Content-Type": "multipart/form-data" } });
      // backend returns { summary, data }
      const s = resp.data.summary || null;
      const d = resp.data.data || resp.data.predictions || [];
      setSummary(s);
      setPredictions(d);
      toast.success("Prediction completed");
    } catch (err: any) {
      toast.error(err.response?.data?.error || "Prediction failed");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const sendSampleReport = async () => {
    const sample = { leaks: [{ zone_id: "Z-1", timestamp: new Date().toISOString(), description: "Sample leak" }] };
    try {
      const resp = await api.post("/report_leak", sample);
      toast.success(resp.data.message || "Report sent");
    } catch (err: any) {
      toast.error(err.response?.data?.error || "Report failed");
    }
  };

  return (
    <div className="space-y-6">
      <Card className="dashboard-section">
        <div className="flex items-start gap-4 mb-4">
          <div>
            <h2 className="text-2xl font-bold">Predict from CSV</h2>
            <p className="text-sm text-muted-foreground">Upload a CSV to run leak predictions using the server model</p>
          </div>
        </div>

        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Heatmap</h3>
          <p className="text-sm text-muted-foreground">Generate a per-zone heatmap (server aggregation) from the uploaded CSV.</p>
          <div className="mt-2 flex gap-2">
            <Button onClick={generateHeatmap} disabled={heatLoading || !file}>
              {heatLoading ? 'Generating...' : 'Generate Heatmap'}
            </Button>
            <Button onClick={() => { setHeatPoints([]); }} variant="ghost">Clear</Button>
          </div>

          {heatPoints.length > 0 && (
            <div className="mt-4" style={{ height: 360 }}>
              <MapContainer style={{ height: '100%', width: '100%' }} center={[heatPoints[0][0], heatPoints[0][1]]} zoom={12} scrollWheelZoom={false}>
                <TileLayer
                  attribution='&copy; OpenStreetMap contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <HeatLayer points={heatPoints} />
              </MapContainer>
            </div>
          )}
        </div>

        <form onSubmit={runPredict} className="space-y-4">
          <div>
            <Label htmlFor="predict-file">CSV File</Label>
            <Input id="predict-file" type="file" accept=".csv" onChange={handleFile} className="mt-2" />
          </div>

          <Button type="submit" className="water-gradient w-full" disabled={loading || !file}>
            {loading ? "Predicting..." : "Run Prediction"}
          </Button>
        </form>

        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Leak Prediction Analysis</h3>
          {predictions.length === 0 ? (
            <div className="text-muted-foreground">No predictions yet</div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-background p-3 rounded" style={{ height: 260 }}>
                  <h4 className="text-sm font-medium mb-2">Leak vs No-Leak (summary)</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={[{ name: "Leaks", value: summary?.leak_count ?? 0 }, { name: "No Leak", value: summary?.no_leak_count ?? 0 }]}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={70}
                        label
                      >
                        <Cell key="c1" fill="#ef4444" />
                        <Cell key="c2" fill="#10b981" />
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="mt-2 text-sm">
                    {summary?.leak_count ?? 0} leak points detected out of {(summary?.leak_count ?? 0) + (summary?.no_leak_count ?? 0)} rows.
                  </div>
                </div>

                <div className="bg-background p-3 rounded" style={{ height: 260 }}>
                  <h4 className="text-sm font-medium mb-2">Leaks by Zone (bar)</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={
                      // build zone counts
                      (() => {
                        const map: Record<string, number> = {};
                        predictions.forEach((r: any) => {
                          const zone = r.zone_id || r.zone || "Unknown";
                          const pred = Number(r.Predicted_Leak ?? r.Leak_Prediction ?? r.predicted ?? r.predicted_leak ?? 0);
                          if (!map[zone]) map[zone] = 0;
                          if (pred === 1) map[zone]++;
                        });
                        return Object.keys(map).map(z => ({ zone: z, leaks: map[z] }));
                      })()
                    }>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="zone" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="leaks" fill="#ef4444" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Usage increase/decrease analysis */}
              <div className="bg-background p-3 rounded">
                <h4 className="text-sm font-medium mb-2">Usage change analysis (water loss)</h4>
                <UsageAnalysisTable data={predictions} />
              </div>

              {/* Data table (sample) */}
              <div className="bg-background p-3 rounded overflow-auto">
                <h4 className="text-sm font-medium mb-2">Predictions (sample rows)</h4>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left border-b">
                      {Object.keys(predictions[0]).slice(0, 12).map(key => (
                        <th key={key} className="p-2">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.slice(0, 25).map((pred, idx) => (
                      <tr key={idx} className="border-b">
                        {Object.values(pred).slice(0, 12).map((val: any, i) => (
                          <td key={i} className="p-2">{typeof val === 'object' ? JSON.stringify(val) : String(val)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Quick Report</h3>
          <p className="text-sm text-muted-foreground">Send a sample leak report to the backend (uses /report_leak)</p>
          <div className="mt-2">
            <Button onClick={sendSampleReport} variant="outline">Send Sample Report</Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PredictAndReport;

function HeatLayer({ points }: { points: Array<[number, number, number, string, number]> }) {
  const map = useMap();
  useEffect(() => {
    if (!map || !points || points.length === 0) return;
    // convert points to [lat, lon, weight]
    const latlngs = points.map(p => [p[0], p[1], p[2]] as [number, number, number]);
    const heat = (L as any).heatLayer(latlngs, { radius: 25, blur: 15, maxZoom: 17 });
    heat.addTo(map);
    try {
      const bounds = L.latLngBounds(latlngs.map((p: any) => [p[0], p[1]]));
      map.fitBounds(bounds.pad(0.2));
    } catch (e) {
      // ignore
    }
    return () => {
      try { map.removeLayer(heat); } catch (e) {}
    };
  }, [map, points]);
  return null;
}
function UsageAnalysisTable({ data }: { data: any[] }) {
  const summary = useMemo(() => {
    // compute water loss per row when possible
    const zoneMap: Record<string, { totalLoss: number; count: number }> = {};
    let overallTotal = 0;
    let overallCount = 0;

    data.forEach((r: any) => {
      const zone = r.zone_id || r.zone || "Unknown";
      const supplied = parseFloat(r.water_supplied_litres ?? r.water_supplied ?? r.supplied ?? NaN);
      const consumed = parseFloat(r.water_consumed_litres ?? r.water_consumed ?? r.consumed ?? NaN);
      if (!isNaN(supplied) && !isNaN(consumed)) {
        const loss = supplied - consumed;
        if (!zoneMap[zone]) zoneMap[zone] = { totalLoss: 0, count: 0 };
        zoneMap[zone].totalLoss += loss;
        zoneMap[zone].count += 1;
        overallTotal += loss;
        overallCount += 1;
      }
    });

    const zones = Object.keys(zoneMap).map((z) => ({ zone: z, avgLoss: zoneMap[z].totalLoss / zoneMap[z].count, count: zoneMap[z].count }));
    const overallAvg = overallCount > 0 ? overallTotal / overallCount : 0;

    const increased = zones.filter(z => z.avgLoss > overallAvg).sort((a,b) => b.avgLoss - a.avgLoss).slice(0,5);
    const decreased = zones.filter(z => z.avgLoss <= overallAvg).sort((a,b) => a.avgLoss - b.avgLoss).slice(0,5);

    return { zones, overallAvg, increased, decreased };
  }, [data]);

  return (
    <div>
      <div className="text-sm mb-2">Overall average water loss per recorded row: {summary.overallAvg.toFixed(2)}</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h5 className="font-medium">Top zones with increased loss</h5>
          {summary.increased.length === 0 ? <div className="text-muted-foreground">No data</div> : (
            <ol className="list-decimal pl-5 mt-2 text-sm">
              {summary.increased.map(z => (
                <li key={z.zone}>{z.zone}: avg loss {z.avgLoss.toFixed(2)} ({z.count} rows)</li>
              ))}
            </ol>
          )}
        </div>

        <div>
          <h5 className="font-medium">Top zones with decreased loss</h5>
          {summary.decreased.length === 0 ? <div className="text-muted-foreground">No data</div> : (
            <ol className="list-decimal pl-5 mt-2 text-sm">
              {summary.decreased.map(z => (
                <li key={z.zone}>{z.zone}: avg loss {z.avgLoss.toFixed(2)} ({z.count} rows)</li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div className="mt-4 text-sm">
        <h5 className="font-medium mb-2">Zone averages (sample)</h5>
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left border-b">
                <th className="p-2">Zone</th>
                <th className="p-2">Avg Loss</th>
                <th className="p-2">Rows</th>
              </tr>
            </thead>
            <tbody>
              {summary.zones.slice(0, 20).map(z => (
                <tr key={z.zone} className="border-b">
                  <td className="p-2">{z.zone}</td>
                  <td className="p-2">{z.avgLoss.toFixed(2)}</td>
                  <td className="p-2">{z.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
