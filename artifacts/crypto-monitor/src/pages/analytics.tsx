// Wired into the router at `/analytics` (see `App.tsx`) and surfaced via
// the sidebar nav in `components/layout.tsx`. Kept as the deep
// agent-performance surface the dashboard summary cards link out to.
import { useState } from "react";
import { useGetAgentPerformance, useGetDashboard, useListAgents, getGetAgentPerformanceQueryKey, getGetDashboardQueryKey } from "@workspace/api-client-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { formatPercentage } from "@/lib/format";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, PieChart, Pie, Cell } from "recharts";
import { cn } from "@/lib/utils";

const AGENT_COLORS = [
  "#00ff9d", "#ff3366", "#00ccff", "#ffcc00", "#ff6600",
  "#cc33ff", "#33ff66", "#ff3333", "#3366ff", "#ff9933",
];

export default function Analytics() {
  const [timeRange, setTimeRange] = useState("24");

  const { data: performance, isLoading: perfLoading } = useGetAgentPerformance(
    { hours: parseInt(timeRange) },
    { query: { refetchInterval: 15000, queryKey: getGetAgentPerformanceQueryKey({ hours: parseInt(timeRange) }) } }
  );
  const { data: dashboard } = useGetDashboard({ query: { refetchInterval: 15000, queryKey: getGetDashboardQueryKey() } });
  const { data: agents } = useListAgents();

  if (perfLoading) {
    return (
      <div className="space-y-6" data-testid="analytics-loading">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-96" />
      </div>
    );
  }

  const agentScoreData = agents?.map((a, i) => ({
    name: a.name,
    score: a.score,
    accuracy: a.accuracy,
    predictions: a.totalPredictions,
    fill: AGENT_COLORS[i % AGENT_COLORS.length],
  })) || [];

  const outcomeData = dashboard ? [
    { name: "Correct", value: dashboard.predictionsByOutcome.correct, fill: "#00ff9d" },
    { name: "Wrong", value: dashboard.predictionsByOutcome.wrong, fill: "#ff3366" },
    { name: "Pending", value: dashboard.predictionsByOutcome.pending, fill: "#ffcc00" },
  ] : [];

  const agentPerformanceByAgent = new Map<number, { name: string; points: { timestamp: string; accuracy: number; score: number }[] }>();
  performance?.forEach((p) => {
    if (!agentPerformanceByAgent.has(p.agentId)) {
      agentPerformanceByAgent.set(p.agentId, { name: p.agentName, points: [] });
    }
    agentPerformanceByAgent.get(p.agentId)!.points.push({
      timestamp: p.timestamp,
      accuracy: p.accuracy,
      score: p.score,
    });
  });

  return (
    <div className="space-y-6" data-testid="analytics-page">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-display font-bold tracking-tight gradient-text" data-testid="text-page-title">Analytics</h1>
          <p className="text-sm text-muted-foreground mt-2">
            Deep performance analysis across all AI agents
          </p>
        </div>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[160px] font-mono text-sm" data-testid="select-time-range">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">Last 1 hour</SelectItem>
            <SelectItem value="6">Last 6 hours</SelectItem>
            <SelectItem value="24">Last 24 hours</SelectItem>
            <SelectItem value="72">Last 3 days</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-scores-chart">
          <CardHeader>
            <CardTitle className="text-sm font-mono uppercase tracking-wider">Agent Scores Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={agentScoreData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={10} angle={-20} textAnchor="end" height={60} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontFamily: "monospace", fontSize: "12px" }}
                  />
                  <Bar dataKey="score" name="Score" radius={[4, 4, 0, 0]}>
                    {agentScoreData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-outcome-distribution">
          <CardHeader>
            <CardTitle className="text-sm font-mono uppercase tracking-wider">Prediction Outcomes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center">
              {outcomeData.some((d) => d.value > 0) ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={outcomeData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={4}
                      dataKey="value"
                    >
                      {outcomeData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} fillOpacity={0.8} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontFamily: "monospace", fontSize: "12px" }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-muted-foreground font-mono text-sm">No data yet</div>
              )}
            </div>
            <div className="flex justify-center gap-6 mt-2">
              {outcomeData.map((d) => (
                <div key={d.name} className="flex items-center gap-2 text-xs font-mono">
                  <span className="w-3 h-3 rounded-full" style={{ background: d.fill }} />
                  {d.name}: {d.value}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-accuracy-comparison">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider">Agent Accuracy Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={agentScoreData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[0, 100]} />
                <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={10} width={120} />
                <Tooltip
                  contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontFamily: "monospace", fontSize: "12px" }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Accuracy"]}
                />
                <Bar dataKey="accuracy" name="Accuracy %" radius={[0, 4, 4, 0]}>
                  {agentScoreData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} fillOpacity={0.7} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-leaderboard">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider">Agent Leaderboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm font-mono" data-testid="table-leaderboard">
              <thead>
                <tr className="text-xs text-muted-foreground uppercase border-b border-border/30">
                  <th className="text-left py-3 px-2">Rank</th>
                  <th className="text-left py-3 px-2">Agent</th>
                  <th className="text-right py-3 px-2">Score</th>
                  <th className="text-right py-3 px-2">Accuracy</th>
                  <th className="text-right py-3 px-2">W/L</th>
                  <th className="text-right py-3 px-2">Total</th>
                  <th className="text-right py-3 px-2">Streak</th>
                </tr>
              </thead>
              <tbody>
                {agents?.map((agent, i) => (
                  <tr key={agent.id} className="border-b border-border/10 hover:bg-white/5 transition-colors" data-testid={`row-agent-${agent.id}`}>
                    <td className="py-3 px-2 text-muted-foreground">#{i + 1}</td>
                    <td className="py-3 px-2 font-medium">{agent.name}</td>
                    <td className="py-3 px-2 text-right font-bold">{agent.score.toFixed(1)}</td>
                    <td className="py-3 px-2 text-right">{formatPercentage(agent.accuracy)}</td>
                    <td className="py-3 px-2 text-right">
                      <span className="text-emerald-400">{agent.correctPredictions}</span>
                      <span className="text-muted-foreground/50">/</span>
                      <span className="text-red-400">{agent.wrongPredictions}</span>
                    </td>
                    <td className="py-3 px-2 text-right text-muted-foreground">{agent.totalPredictions}</td>
                    <td className="py-3 px-2 text-right">
                      {agent.streak > 0 ? (
                        <span className={cn(agent.streakType === "win" ? "text-emerald-400" : "text-red-400")}>
                          {agent.streak}x {agent.streakType}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
