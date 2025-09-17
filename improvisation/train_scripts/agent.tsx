import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Pause, Play, RefreshCw, Send, Sparkles } from "lucide-react";
import React, { useMemo, useState } from "react";
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

// ------------------------------
// 型定義
// ------------------------------
interface AgentConfig {
  name: string;
  Rself: number; // 自己責任の強度
  n: number; // 熟練度
  inertia: number; // 慣性（直前方向を引きずる度合い）
  theta: number; // 臨界閾値
  weights: { r: number; n: number; k: number; e: number; u: number; nov: number };
}

interface TurnState {
  lastDirection: [number, number];
  lastMsg: string;
  t: number;
}

interface Candidate {
  dir: [number, number];
  text: string;
}

interface ScoreBreakdown {
  r: number; n: number; k: number; e: number; u: number; nov: number; eps: number; total: number;
}

interface LogRow {
  turn: number;
  speaker: string;
  incoming: string;
  decided: boolean;
  bestText: string;
  score: number;
  theta: number;
  Rdot: number;
  K: number;
  E: number;
  U: number;
  Novel: number;
}

// ------------------------------
// ユーティリティ
// ------------------------------
const dot = (a: [number, number], b: [number, number]) => a[0] * b[0] + a[1] * b[1];
const norm = (v: [number, number]) => Math.sqrt(v[0] * v[0] + v[1] * v[1]) || 1;
const normalize = (v: [number, number]): [number, number] => {
  const n = norm(v);
  return [v[0] / n, v[1] / n];
};

function gauss(mean = 0, std = 1) {
  // Box–Muller
  let u = 1 - Math.random();
  let v = 1 - Math.random();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * std + mean;
}

function clamp(x: number, lo: number, hi: number) { return Math.min(Math.max(x, lo), hi); }

function extractEntropyFeatures(msg: string) {
  const abstractWords = ["意味", "未来", "可能性", "なぜ", "どうして", "責任", "Stillness", "矛盾", "臨界"];
  const abstract = abstractWords.reduce((acc, k) => acc + (msg.includes(k) ? 1 : 0), 0);
  const E = 0.2 * Math.min(msg.length, 400) / 100 + 0.5 * (msg.includes("? ") || msg.endsWith("?" ) ? 1 : 0) + 0.3 * (msg.includes("!") ? 1 : 0) + 0.2 * abstract;
  return { E: clamp(E, 0, 2.0), novBias: 0.1 * abstract };
}

function genCandidates(incoming: string): Candidate[] {
  // 方向性の例（情報/共感/挑発）
  const dirs: [number, number][] = [ [1, 0], [0.7, 0.7], [0, 1] ];
  const texts = [
    "要点を整理して次の実験手順を提案する。",
    "感情の行間を汲みつつ確認質問を返す。",
    "新しい視点を持ち込んで小さく挑発する。",
  ];
  return dirs.map((d, i) => ({ dir: normalize(d), text: texts[i] }));
}

function scoreCandidate(cfg: AgentConfig, ctx: TurnState, cand: Candidate, feats: { E: number; novBias: number }): ScoreBreakdown {
  const Rdot = cfg.Rself * dot(cand.dir, [1, 0]); // Rselfはx方向に張る例
  const K = cfg.inertia * dot(cand.dir, ctx.lastDirection);
  const E = feats.E;
  const U = cand.text.includes("提案") ? 0.7 : 0.5;
  const Novel = feats.novBias + (cand.text.includes("新しい") ? 0.3 : 0.1);
  const eps = gauss(0, 0.05);
  const total = cfg.weights.r * Rdot + cfg.weights.n * cfg.n + cfg.weights.k * K + cfg.weights.e * E + cfg.weights.u * U + cfg.weights.nov * Novel + eps;
  return { r: cfg.weights.r * Rdot, n: cfg.weights.n * cfg.n, k: cfg.weights.k * K, e: cfg.weights.e * E, u: cfg.weights.u * U, nov: cfg.weights.nov * Novel, eps, total };
}

function decide(cfg: AgentConfig, ctx: TurnState, incoming: string) {
  const feats = extractEntropyFeatures(incoming);
  const cands = genCandidates(incoming);
  const scored = cands.map(c => ({ c, s: scoreCandidate(cfg, ctx, c, feats) }))
                      .sort((a, b) => b.s.total - a.s.total);
  const best = scored[0];
  const decided = best.s.total >= cfg.theta;
  const breakdown = best.s;
  return { decided, best, breakdown, feats, all: scored };
}

function update(ctx: TurnState, decided: boolean, chosen?: Candidate) {
  if (decided && chosen) ctx.lastDirection = chosen.dir; else ctx.lastDirection = [ctx.lastDirection[0] * 0.95, ctx.lastDirection[1] * 0.95];
  ctx.t += 1;
}

// ------------------------------
// UI コンポーネント
// ------------------------------
function ParamSlider({ label, value, onChange, min = 0, max = 1, step = 0.01 }:{ label:string; value:number; onChange:(v:number)=>void; min?:number; max?:number; step?:number }){
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{value.toFixed(2)}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={(v)=>onChange(v[0])} />
    </div>
  );
}

function WeightSliders({ cfg, setCfg }:{ cfg: AgentConfig; setCfg: (u: Partial<AgentConfig>)=>void }){
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      <ParamSlider label="w_r (責任方向)" value={cfg.weights.r} onChange={(v)=>setCfg({ weights: { ...cfg.weights, r: v } })} />
      <ParamSlider label="w_n (熟練度)" value={cfg.weights.n} onChange={(v)=>setCfg({ weights: { ...cfg.weights, n: v } })} />
      <ParamSlider label="w_k (慣性)" value={cfg.weights.k} onChange={(v)=>setCfg({ weights: { ...cfg.weights, k: v } })} />
      <ParamSlider label="w_e (外部圧E)" value={cfg.weights.e} onChange={(v)=>setCfg({ weights: { ...cfg.weights, e: v } })} />
      <ParamSlider label="w_u (有用性)" value={cfg.weights.u} onChange={(v)=>setCfg({ weights: { ...cfg.weights, u: v } })} />
      <ParamSlider label="w_nov (新規性)" value={cfg.weights.nov} onChange={(v)=>setCfg({ weights: { ...cfg.weights, nov: v } })} />
    </div>
  );
}

function AgentCard({ title, cfg, setCfg }:{ title:string; cfg:AgentConfig; setCfg:(u: Partial<AgentConfig>)=>void }){
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader className="pb-3"><CardTitle className="text-lg">{title}</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <ParamSlider label="Rself" value={cfg.Rself} onChange={(v)=>setCfg({ Rself: v })} />
          <ParamSlider label="熟練度 n" value={cfg.n} onChange={(v)=>setCfg({ n: v })} />
          <ParamSlider label="慣性 inertia" value={cfg.inertia} onChange={(v)=>setCfg({ inertia: v })} />
          <ParamSlider label="閾値 θ" value={cfg.theta} onChange={(v)=>setCfg({ theta: v })} min={0.5} max={3} />
        </div>
        <WeightSliders cfg={cfg} setCfg={setCfg} />
      </CardContent>
    </Card>
  );
}

// ------------------------------
// メインアプリ
// ------------------------------
export default function WCPApp(){
  const [cfgA, setCfgA] = useState<AgentConfig>({
    name: "A",
    Rself: 0.9,
    n: 0.8,
    inertia: 0.6,
    theta: 1.6,
    weights: { r: 0.7, n: 0.3, k: 0.5, e: 0.6, u: 0.6, nov: 0.4 },
  });
  const [cfgB, setCfgB] = useState<AgentConfig>({
    name: "B",
    Rself: 0.7,
    n: 0.6,
    inertia: 0.5,
    theta: 1.4,
    weights: { r: 0.6, n: 0.3, k: 0.4, e: 0.7, u: 0.5, nov: 0.5 },
  });

  const [ctxA, setCtxA] = useState<TurnState>({ lastDirection: [0,0], lastMsg: "", t: 0 });
  const [ctxB, setCtxB] = useState<TurnState>({ lastDirection: [0,0], lastMsg: "", t: 0 });

  const [aFirstMsg, setAFirstMsg] = useState("意思発生を“臨界”として扱うなら、いつ越える？");
  const [autoLoop, setAutoLoop] = useState(false);
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [turnLimit, setTurnLimit] = useState(12);

  // 1ステップ（A→B）
  const stepOnce = () => {
    let incomingForA = logs.length === 0 ? aFirstMsg : logs[logs.length - 1].bestText || "（保留…）";

    // A が考える
    const decA = decide(cfgA, ctxA, incomingForA);
    const chosenA = decA.best.c;
    const decidedA = decA.decided;
    const outA = decidedA ? chosenA.text : "（保留：さらに内省中…）";

    const newLogA: LogRow = {
      turn: ctxA.t + 1,
      speaker: "A",
      incoming: incomingForA,
      decided: decidedA,
      bestText: outA,
      score: decA.breakdown.total,
      theta: cfgA.theta,
      Rdot: decA.breakdown.r,
      K: decA.breakdown.k,
      E: decA.breakdown.e,
      U: decA.breakdown.u,
      Novel: decA.breakdown.nov,
    };

    const newCtxA: TurnState = { ...ctxA };
    update(newCtxA, decidedA, chosenA);

    // B に届く
    const decB = decide(cfgB, ctxB, outA);
    const chosenB = decB.best.c;
    const decidedB = decB.decided;
    const outB = decidedB ? chosenB.text : "（保留と思索を続ける…）";

    const newLogB: LogRow = {
      turn: ctxB.t + 1,
      speaker: "B",
      incoming: outA,
      decided: decidedB,
      bestText: outB,
      score: decB.breakdown.total,
      theta: cfgB.theta,
      Rdot: decB.breakdown.r,
      K: decB.breakdown.k,
      E: decB.breakdown.e,
      U: decB.breakdown.u,
      Novel: decB.breakdown.nov,
    };

    const newCtxB: TurnState = { ...ctxB };
    update(newCtxB, decidedB, chosenB);

    setCtxA(newCtxA);
    setCtxB(newCtxB);
    setLogs(prev => [...prev, newLogA, newLogB].slice(-200));
  };

  // オートループ
  React.useEffect(() => {
    if (!autoLoop) return;
    if (logs.length / 2 >= turnLimit) { setAutoLoop(false); return; }
    const id = setTimeout(stepOnce, 400);
    return () => clearTimeout(id);
  }, [autoLoop, logs, turnLimit]);

  const resetAll = () => {
    setCtxA({ lastDirection: [0,0], lastMsg: "", t: 0 });
    setCtxB({ lastDirection: [0,0], lastMsg: "", t: 0 });
    setLogs([]);
  };

  const chartData = useMemo(() => logs.map((l, i) => ({
    idx: i + 1,
    score: Number(l.score.toFixed(3)),
    theta: Number(l.theta.toFixed(3)),
    E: Number(l.E.toFixed(3)),
    U: Number(l.U.toFixed(3)),
    Novel: Number(l.Novel.toFixed(3)),
    K: Number(l.K.toFixed(3)),
    Rdot: Number(l.Rdot.toFixed(3)),
    speaker: l.speaker,
  })), [logs]);

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-white to-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight flex items-center gap-2">
            <Sparkles className="w-6 h-6"/> 文通式「意思発生」WCP
          </h1>
          <div className="flex items-center gap-2">
            <Button variant={autoLoop ? "secondary" : "default"} onClick={()=>setAutoLoop(v=>!v)}>
              {autoLoop ? <Pause className="mr-2 h-4 w-4"/> : <Play className="mr-2 h-4 w-4"/>}
              {autoLoop ? "停止" : "自動進行"}
            </Button>
            <Button variant="outline" onClick={stepOnce}><Send className="mr-2 h-4 w-4"/>1ターン進める</Button>
            <Button variant="ghost" onClick={resetAll}><RefreshCw className="mr-2 h-4 w-4"/>リセット</Button>
          </div>
        </header>

        <Card className="rounded-2xl">
          <CardHeader className="pb-3"><CardTitle className="text-lg">初手メッセージ（A→B）</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Textarea value={aFirstMsg} onChange={(e)=>setAFirstMsg(e.target.value)} placeholder="最初の往復を始める文を入力" />
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-3">
                <Label htmlFor="turnLimit" className="text-sm text-muted-foreground">最大ターン（往復数）</Label>
                <Input id="turnLimit" type="number" min={1} max={100} value={turnLimit} onChange={(e)=>setTurnLimit(Number(e.target.value))} className="w-28"/>
              </div>
              <div className="flex items-center gap-3">
                <Switch id="auto" checked={autoLoop} onCheckedChange={setAutoLoop} />
                <Label htmlFor="auto">自動進行を有効化</Label>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AgentCard title="Agent A（君の分身）" cfg={cfgA} setCfg={(u)=>setCfgA(prev=>({ ...prev, ...u, weights: u.weights ? u.weights : prev.weights }))} />
          <AgentCard title="Agent B（相手/AI）" cfg={cfgB} setCfg={(u)=>setCfgB(prev=>({ ...prev, ...u, weights: u.weights ? u.weights : prev.weights }))} />
        </div>

        <Card className="rounded-2xl">
          <CardHeader className="pb-3"><CardTitle className="text-lg">スコア・臨界の可視化</CardTitle></CardHeader>
          <CardContent>
            <div className="h-72 w-full">
              <ResponsiveContainer>
                <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="idx" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="score" name="score(合計)" dot={false} />
                  <Line type="monotone" dataKey="theta" name="θ(閾値)" strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="E" name="E(外部圧)" dot={false} />
                  <Line type="monotone" dataKey="K" name="K(慣性)" dot={false} />
                  <Line type="monotone" dataKey="Rdot" name="Rdot(責任方向)" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl">
          <CardHeader className="pb-3"><CardTitle className="text-lg">ログ（どの力学で意思が生まれたか）</CardTitle></CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">話者</th>
                  <th className="py-2 pr-3">incoming</th>
                  <th className="py-2 pr-3">bestText</th>
                  <th className="py-2 pr-3">score</th>
                  <th className="py-2 pr-3">θ</th>
                  <th className="py-2 pr-3">Rdot</th>
                  <th className="py-2 pr-3">K</th>
                  <th className="py-2 pr-3">E</th>
                  <th className="py-2 pr-3">U</th>
                  <th className="py-2 pr-3">Novel</th>
                  <th className="py-2 pr-3">決定</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((l, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-2 pr-3 font-mono">{l.turn}</td>
                    <td className="py-2 pr-3">{l.speaker}</td>
                    <td className="py-2 pr-3 max-w-[22rem] truncate" title={l.incoming}>{l.incoming}</td>
                    <td className="py-2 pr-3 max-w-[22rem] truncate" title={l.bestText}>{l.bestText}</td>
                    <td className="py-2 pr-3 font-mono">{l.score.toFixed(3)}</td>
                    <td className="py-2 pr-3 font-mono">{l.theta.toFixed(2)}</td>
                    <td className="py-2 pr-3 font-mono">{l.Rdot.toFixed(3)}</td>
                    <td className="py-2 pr-3 font-mono">{l.K.toFixed(3)}</td>
                    <td className="py-2 pr-3 font-mono">{l.E.toFixed(3)}</td>
                    <td className="py-2 pr-3 font-mono">{l.U.toFixed(3)}</td>
                    <td className="py-2 pr-3 font-mono">{l.Novel.toFixed(3)}</td>
                    <td className="py-2 pr-3">{l.decided ? "✅" : "…"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <footer className="text-xs text-muted-foreground text-center py-6">
          <p>© 2025 WCP: Wai Correspondence Protocol — Stillness → Motion 臨界で“意思発生”を観測する。</p>
        </footer>
      </div>
    </div>
  );
}
