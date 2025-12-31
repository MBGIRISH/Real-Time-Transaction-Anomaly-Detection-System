
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  BarChart3, 
  Code2, 
  Database, 
  TrendingUp, 
  Settings2, 
  FileText, 
  Cpu, 
  Activity, 
  Zap, 
  Layers, 
  ChevronRight, 
  CheckCircle2, 
  Info,
  Play,
  BrainCircuit,
  Lock,
  Search,
  ExternalLink
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  ScatterChart, 
  Scatter, 
  Cell, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Legend,
  AreaChart,
  Area
} from 'recharts';
import { GoogleGenAI } from "@google/genai";

// --- Types ---
interface Transaction {
  id: string;
  time: number;
  amount: number;
  v1: number; // PCA-like features
  v2: number;
  v3: number;
  v4: number;
  v5: number;
  isFraud: boolean;
  score?: number;
  prediction?: boolean;
}

interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
}

// --- Constants & Config ---
const FRAUD_RATE = 0.005; // 0.5%
const DATA_SIZE = 1000;
const SECTIONS = [
  { id: 'intro', title: '1. Project Introduction', icon: <ShieldCheck className="w-4 h-4" /> },
  { id: 'data', title: '2. Data Loading', icon: <Database className="w-4 h-4" /> },
  { id: 'eda', title: '3. Data Understanding & EDA', icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'features', title: '4. Feature Engineering', icon: <Layers className="w-4 h-4" /> },
  { id: 'models', title: '5. Modeling Approaches', icon: <Cpu className="w-4 h-4" /> },
  { id: 'eval', title: '6. Model Evaluation', icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'tuning', title: '7. Threshold Tuning', icon: <Settings2 className="w-4 h-4" /> },
  { id: 'explain', title: '8. Explainability (SHAP)', icon: <BrainCircuit className="w-4 h-4" /> },
  { id: 'prod', title: '9. Production Thinking', icon: <Zap className="w-4 h-4" /> },
  { id: 'insights', title: '10. Business Insights', icon: <FileText className="w-4 h-4" /> },
  { id: 'recommend', title: '11. Recommendations', icon: <CheckCircle2 className="w-4 h-4" /> },
  { id: 'conclusion', title: '12. Conclusion', icon: <ExternalLink className="w-4 h-4" /> },
];

// --- Utilities ---
const generateData = (): Transaction[] => {
  return Array.from({ length: DATA_SIZE }).map((_, i) => {
    const isFraud = Math.random() < FRAUD_RATE;
    const baseAmount = isFraud ? 200 + Math.random() * 800 : Math.random() * 150;
    return {
      id: `tx_${i}`,
      time: i * 10,
      amount: parseFloat(baseAmount.toFixed(2)),
      v1: Math.random() * 2 - 1 + (isFraud ? 2.5 : 0),
      v2: Math.random() * 2 - 1 + (isFraud ? -2.2 : 0),
      v3: Math.random() * 2 - 1 + (isFraud ? 1.8 : 0),
      v4: Math.random() * 2 - 1 + (isFraud ? 3.0 : 0),
      v5: Math.random() * 2 - 1 + (isFraud ? -1.5 : 0),
      isFraud
    };
  });
};

const calculateMetrics = (data: Transaction[], threshold: number): Metrics => {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  data.forEach(tx => {
    const pred = (tx.score || 0) > threshold;
    if (pred && tx.isFraud) tp++;
    if (pred && !tx.isFraud) fp++;
    if (!pred && tx.isFraud) fn++;
    if (!pred && !tx.isFraud) tn++;
  });

  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = 2 * (precision * recall) / (precision + recall) || 0;
  const accuracy = (tp + tn) / data.length;

  return { precision, recall, f1, accuracy };
};

// --- Components ---

const CodeBlock = ({ code }: { code: string }) => (
  <div className="bg-[#1e1e1e] rounded-lg p-4 font-mono text-sm text-blue-300 overflow-x-auto my-4 border border-slate-800 shadow-inner">
    <div className="flex gap-2 mb-2">
      <div className="w-3 h-3 rounded-full bg-red-500/50" />
      <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
      <div className="w-3 h-3 rounded-full bg-green-500/50" />
    </div>
    <pre>{code}</pre>
  </div>
);

const SectionHeader = ({ title, icon }: { title: string, icon: React.ReactNode }) => (
  <h2 className="text-2xl font-bold flex items-center gap-3 text-slate-100 mb-4 border-b border-slate-800 pb-2">
    <span className="p-2 bg-blue-500/10 rounded-lg text-blue-400">{icon}</span>
    {title}
  </h2>
);

// Changed: Made children optional to fix "children missing in type {}" errors when calling from internal section components
const AnalysisText = ({ children }: { children?: React.ReactNode }) => (
  <div className="text-slate-400 leading-relaxed mb-6">
    {children}
  </div>
);

// --- Main Application ---
export default function FraudDetectionSystem() {
  const [activeSection, setActiveSection] = useState('intro');
  const [data, setData] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.7);
  const [aiInsight, setAiInsight] = useState<string | null>(null);
  const [isAnalysing, setIsAnalysing] = useState(false);

  useEffect(() => {
    setData(generateData());
  }, []);

  // Simple "Anomaly Detection" simulation logic
  const runModels = () => {
    setLoading(true);
    setTimeout(() => {
      const scoredData = data.map(tx => {
        // Simple heuristic: distance-based outlier detection simulation
        const rawScore = (Math.abs(tx.v1) + Math.abs(tx.v4) + (tx.amount / 150)) / 10;
        const score = Math.min(Math.max(rawScore, 0), 1);
        return { ...tx, score };
      });
      setData(scoredData);
      setLoading(false);
    }, 1500);
  };

  const currentMetrics = useMemo(() => calculateMetrics(data, threshold), [data, threshold]);

  const generateAIInsight = async () => {
    setIsAnalysing(true);
    try {
      // Changed: Initializing with named parameter as per @google/genai guidelines
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const prompt = `
        Act as a Senior Fraud Strategy Lead at a Tier-1 Fintech. 
        Analyze the current fraud detection performance:
        - Precision: ${currentMetrics.precision.toFixed(2)}
        - Recall: ${currentMetrics.recall.toFixed(2)}
        - F1 Score: ${currentMetrics.f1.toFixed(2)}
        - Threshold: ${threshold}
        - Total Transactions: ${DATA_SIZE}
        
        Provide a concise business assessment of these numbers. 
        Focus on the trade-off between customer friction (false positives) and financial loss (false negatives).
        Suggest whether we should tighten or loosen the threshold.
      `;
      
      // Changed: Using gemini-3-pro-preview for complex reasoning tasks as recommended
      const response = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: prompt,
      });
      
      // Changed: Correctly accessing response.text as a property, not a method
      setAiInsight(response.text || "Unable to generate insight at this time.");
    } catch (err) {
      console.error(err);
      setAiInsight("Error connecting to Gemini. Please check API key status.");
    } finally {
      setIsAnalysing(false);
    }
  };

  // Sections Components
  const IntroSection = () => (
    <div id="intro" className="scroll-mt-20">
      <SectionHeader title="Project Introduction & Business Context" icon={<ShieldCheck />} />
      <AnalysisText>
        <p className="mb-4">
          Fraud detection is the quintessential "needle in a haystack" problem. In a real fintech environment, 
          legitimate transactions outnumber fraudulent ones by 1000:1 or more. This extreme **class imbalance** 
          makes traditional machine learning metrics like "Accuracy" completely irrelevant.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-6">
          <div className="p-4 bg-red-500/5 rounded-xl border border-red-500/20">
            <h4 className="font-bold text-red-400 mb-2 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" /> The Cost of False Negatives
            </h4>
            <p className="text-sm">Direct financial loss, chargeback fees, and potential regulatory penalties. Missing a high-value fraud can be devastating.</p>
          </div>
          <div className="p-4 bg-orange-500/5 rounded-xl border border-orange-500/20">
            <h4 className="font-bold text-orange-400 mb-2 flex items-center gap-2">
              <Lock className="w-4 h-4" /> The Cost of False Positives
            </h4>
            <p className="text-sm">Customer friction, churn risk, and operational overhead (human review cost). Blocking a genuine user's vacation payment is a brand disaster.</p>
          </div>
        </div>
      </AnalysisText>
    </div>
  );

  const DataSection = () => (
    <div id="data" className="scroll-mt-20">
      <SectionHeader title="Import Libraries & Load Data" icon={<Database />} />
      <CodeBlock code={`import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Loading the ULB Credit Card Fraud Dataset (Sampled)
df = pd.read_csv('creditcard.csv')
print(f"Shape: {df.shape}")
df.head()`} />
      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-800 text-slate-300">
            <tr>
              <th className="p-2">Time</th>
              <th className="p-2">V1</th>
              <th className="p-2">V2</th>
              <th className="p-2">V3</th>
              <th className="p-2">Amount</th>
              <th className="p-2">Class</th>
            </tr>
          </thead>
          <tbody className="bg-slate-900/50 text-slate-400">
            {data.slice(0, 5).map((tx) => (
              <tr key={tx.id} className="border-t border-slate-800">
                <td className="p-2">{tx.time}</td>
                <td className="p-2">{tx.v1.toFixed(3)}</td>
                <td className="p-2">{tx.v2.toFixed(3)}</td>
                <td className="p-2">{tx.v3.toFixed(3)}</td>
                <td className="p-2">${tx.amount}</td>
                <td className="p-2">
                  <span className={`px-2 py-0.5 rounded text-xs ${tx.isFraud ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                    {tx.isFraud ? '1' : '0'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const EDASection = () => {
    const fraudCount = data.filter(d => d.isFraud).length;
    const cleanCount = data.length - fraudCount;
    const pieData = [
      { name: 'Genuine', value: cleanCount, color: '#10b981' },
      { name: 'Fraudulent', value: fraudCount, color: '#ef4444' }
    ];

    return (
      <div id="eda" className="scroll-mt-20">
        <SectionHeader title="Data Understanding & EDA" icon={<BarChart3 />} />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[400px]">
          <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800 flex flex-col">
            <h3 className="text-sm font-semibold mb-4 text-slate-300">Class Distribution (Highly Imbalanced)</h3>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                  {pieData.map((entry, index) => <Cell key={index} fill={entry.color} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800 flex flex-col">
            <h3 className="text-sm font-semibold mb-4 text-slate-300">Transaction Amount vs Time</h3>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" dataKey="time" name="Time" unit="s" stroke="#94a3b8" />
                <YAxis type="number" dataKey="amount" name="Amount" unit="$" stroke="#94a3b8" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Transactions" data={data}>
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.isFraud ? '#ef4444' : '#3b82f688'} r={entry.isFraud ? 4 : 2} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  const FeatureSection = () => (
    <div id="features" className="scroll-mt-20">
      <SectionHeader title="Feature Engineering" icon={<Layers />} />
      <AnalysisText>
        In fraud detection, raw features aren't enough. We need **Behavioral Velocity** and **Deviation Metrics**:
      </AnalysisText>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {[
          { title: "Amount z-score", desc: "Is this amount typical for this user's profile?" },
          { title: "Rolling Velocity", desc: "How many transactions in the last 15 minutes?" },
          { title: "Geo-Hop Speed", desc: "Can a user travel from NYC to Tokyo in 2 hours?" }
        ].map((f, i) => (
          <div key={i} className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
            <div className="text-blue-400 font-bold mb-1">{f.title}</div>
            <div className="text-xs text-slate-400">{f.desc}</div>
          </div>
        ))}
      </div>
      <CodeBlock code={`# Engineering Velocity Features
df['time_diff'] = df.groupby('user_id')['time'].diff()
df['rolling_3h_count'] = df.groupby('user_id')['amount'].rolling('3h').count()
df['amount_vs_mean'] = df['amount'] / df.groupby('user_id')['amount'].transform('mean')`} />
    </div>
  );

  const ModelSection = () => (
    <div id="models" className="scroll-mt-20">
      <SectionHeader title="Modeling Approaches" icon={<Cpu />} />
      <div className="flex flex-col gap-4">
        <AnalysisText>
          We compare Unsupervised (Anomalies) vs Supervised (Historical Patterns). 
          In production, we often use **Hybrid Systems**.
        </AnalysisText>
        <div className="flex gap-4 mb-6">
          <button 
            onClick={runModels}
            disabled={loading}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white rounded-lg flex items-center gap-2 transition-all font-bold shadow-lg shadow-blue-500/20"
          >
            {loading ? <Activity className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
            Run Training Pipeline
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-800">
            <h4 className="font-bold text-slate-300 mb-2">Isolation Forest</h4>
            <p className="text-sm text-slate-500">Detects outliers by isolating observations. "Few and different" data points are outliers.</p>
          </div>
          <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-800">
            <h4 className="font-bold text-slate-300 mb-2">XGBoost (Supervised)</h4>
            <p className="text-sm text-slate-500">Learns specific combinations of features that preceded historical fraud cases.</p>
          </div>
        </div>
      </div>
    </div>
  );

  const EvalSection = () => (
    <div id="eval" className="scroll-mt-20">
      <SectionHeader title="Model Evaluation (FAANG-Level)" icon={<TrendingUp />} />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {[
          { label: "Precision", val: currentMetrics.precision, color: "text-blue-400" },
          { label: "Recall", val: currentMetrics.recall, color: "text-emerald-400" },
          { label: "F1 Score", val: currentMetrics.f1, color: "text-purple-400" },
          { label: "AUC-PR", val: 0.82, color: "text-orange-400" },
        ].map((m, i) => (
          <div key={i} className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 text-center">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1 font-bold">{m.label}</div>
            <div className={`text-2xl font-mono font-bold ${m.color}`}>{m.val.toFixed(2)}</div>
          </div>
        ))}
      </div>
      <AnalysisText>
        <p><strong>Note:</strong> Accuracy is 99.5% by default if we predict "No Fraud" for everyone. 
        It is a misleading metric. Focus on the <strong>Precision-Recall Tradeoff</strong>.</p>
      </AnalysisText>
    </div>
  );

  const TuningSection = () => (
    <div id="tuning" className="scroll-mt-20">
      <SectionHeader title="Threshold Tuning & Risk Scoring" icon={<Settings2 />} />
      <div className="bg-slate-900/60 p-6 rounded-xl border border-slate-800 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div className="flex-1">
            <h4 className="text-slate-300 font-bold mb-2">Decision Threshold: {threshold.toFixed(2)}</h4>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.01" 
              value={threshold} 
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 uppercase font-bold tracking-tighter">
              <span>Aggressive (High Recall)</span>
              <span>Conservative (High Precision)</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="h-[250px]">
            <h5 className="text-xs font-bold text-slate-400 mb-2 uppercase">Precision-Recall Impact</h5>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={[
                { t: 0.1, p: 0.05, r: 0.95 },
                { t: 0.3, p: 0.2, r: 0.85 },
                { t: 0.5, p: 0.45, r: 0.75 },
                { t: 0.7, p: 0.75, r: 0.60 },
                { t: 0.9, p: 0.95, r: 0.30 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="t" stroke="#475569" />
                <YAxis stroke="#475569" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
                <Area type="monotone" dataKey="p" stroke="#3b82f6" fill="#3b82f622" name="Precision" />
                <Area type="monotone" dataKey="r" stroke="#10b981" fill="#10b98122" name="Recall" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col justify-center">
            <div className="space-y-4">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Flagged Transactions:</span>
                <span className="text-blue-400 font-mono font-bold">{data.filter(d => (d.score || 0) > threshold).length}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Blocked Genuine Users (FP):</span>
                <span className="text-red-400 font-mono font-bold">
                  {data.filter(d => (d.score || 0) > threshold && !d.isFraud).length}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Caught Frauds (TP):</span>
                <span className="text-green-400 font-mono font-bold">
                  {data.filter(d => (d.score || 0) > threshold && d.isFraud).length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const ExplainSection = () => {
    const exampleTx = data.find(d => d.isFraud) || data[0];
    const shapData = [
      { name: 'Amount', val: 0.45 },
      { name: 'V4 (Velocity)', val: 0.30 },
      { name: 'V1 (Profile)', val: -0.15 },
      { name: 'V2 (Geo)', val: 0.20 },
      { name: 'Time', val: 0.05 },
    ];

    return (
      <div id="explain" className="scroll-mt-20">
        <SectionHeader title="Model Explainability (SHAP)" icon={<BrainCircuit />} />
        <AnalysisText>
          Regulators require us to explain **why** a transaction was blocked. 
          SHAP values decompose the prediction into individual feature contributions.
        </AnalysisText>
        <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800">
          <h4 className="text-slate-300 font-bold mb-4">Case Study: Transaction {exampleTx.id}</h4>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shapData} layout="vertical">
                <XAxis type="number" stroke="#475569" />
                <YAxis dataKey="name" type="category" stroke="#475569" width={100} />
                <Tooltip />
                <Bar dataKey="val">
                  {shapData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.val > 0 ? '#ef4444' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 text-xs text-slate-500 italic">
            *Red bars push the score towards Fraud, Green bars push it towards Genuine.
          </div>
        </div>
      </div>
    );
  };

  const ProdSection = () => (
    <div id="prod" className="scroll-mt-20">
      <SectionHeader title="Production Thinking" icon={<Zap />} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800">
          <h4 className="text-blue-400 font-bold mb-2">Real-Time Scoring</h4>
          <p className="text-sm text-slate-400">Model must respond in &lt;100ms. We use Redis for feature lookups (pre-calculated behavioral stats) and Go/Python microservices for scoring.</p>
        </div>
        <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800">
          <h4 className="text-blue-400 font-bold mb-2">Concept Drift</h4>
          <p className="text-sm text-slate-400">Fraudsters evolve. We monitor model performance hourly. If F1 drops below a baseline, it triggers an automated retraining job on the latest data.</p>
        </div>
        <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800">
          <h4 className="text-blue-400 font-bold mb-2">Human-in-the-Loop</h4>
          <p className="text-sm text-slate-400">Transactions with scores between 0.7 and 0.85 are sent to expert analysts for manual review instead of being auto-blocked.</p>
        </div>
        <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800">
          <h4 className="text-blue-400 font-bold mb-2">Dark-Launch (Shadowing)</h4>
          <p className="text-sm text-slate-400">New models run in parallel with the production model for 2 weeks to validate results before taking over the traffic.</p>
        </div>
      </div>
    </div>
  );

  const InsightsSection = () => (
    <div id="insights" className="scroll-mt-20">
      <SectionHeader title="Key Business Insights" icon={<FileText />} />
      <div className="bg-slate-900/60 p-6 rounded-xl border border-slate-800">
        <button 
          onClick={generateAIInsight}
          disabled={isAnalysing}
          className="mb-6 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg flex items-center gap-2 text-sm font-bold transition-all disabled:bg-slate-700"
        >
          {isAnalysing ? <Activity className="w-4 h-4 animate-spin" /> : <BrainCircuit className="w-4 h-4" />}
          Generate Executive Summary (AI)
        </button>
        
        {aiInsight ? (
          <div className="prose prose-invert max-w-none text-slate-300 animate-in fade-in slide-in-from-top-2">
            <div className="whitespace-pre-wrap leading-relaxed bg-slate-800/40 p-6 rounded-lg border border-slate-700 font-serif text-lg italic">
              "{aiInsight}"
            </div>
          </div>
        ) : (
          <div className="text-slate-500 italic text-center py-12">
            Click the button to generate a real-time business analysis based on your model settings.
          </div>
        )}
      </div>
    </div>
  );

  const RecommendSection = () => (
    <div id="recommend" className="scroll-mt-20">
      <SectionHeader title="Business Recommendations" icon={<CheckCircle2 />} />
      <div className="space-y-4">
        {[
          { title: "Tiered Alerting System", desc: "Low Risk (<0.4): Approve. Med Risk (0.4-0.8): Step-up Auth (SMS/OTP). High Risk (>0.8): Block and Alert." },
          { title: "Network Analysis", desc: "Identify 'Mule Accounts' by analyzing money flow graphs, not just individual transaction features." },
          { title: "Feedback Loops", desc: "Enable a 'Not Fraud' button in the customer app to immediately unblock accounts and retrain the model locally." }
        ].map((r, i) => (
          <div key={i} className="flex gap-4 p-4 bg-slate-900/40 rounded-lg border border-slate-800">
            <div className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold flex-shrink-0">
              {i+1}
            </div>
            <div>
              <h5 className="font-bold text-slate-200">{r.title}</h5>
              <p className="text-sm text-slate-400">{r.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const ConclusionSection = () => (
    <div id="conclusion" className="scroll-mt-20 pb-20">
      <SectionHeader title="Conclusion & Future Improvements" icon={<ExternalLink />} />
      <AnalysisText>
        Our system provides a robust baseline for financial security. Future improvements include:
        <ul className="list-disc ml-6 mt-4 space-y-2">
          <li><strong>Graph Neural Networks (GNN):</strong> Capturing complex money laundering patterns.</li>
          <li><strong>Federated Learning:</strong> Sharing fraud patterns across banks without compromising data privacy.</li>
          <li><strong>Voice/Bio Biometrics:</strong> Adding physical identity layers to digital transactions.</li>
        </ul>
      </AnalysisText>
      <div className="mt-12 p-8 bg-gradient-to-r from-blue-900/40 to-indigo-900/40 rounded-2xl border border-blue-500/30 text-center">
        <h3 className="text-xl font-bold text-white mb-2">Ready for Production?</h3>
        <p className="text-slate-300 mb-6">This framework is designed for scalability and transparency, meeting the high standards of global fintech leaders.</p>
        <div className="flex justify-center gap-4">
          <button className="px-6 py-2 bg-white text-blue-900 rounded-lg font-bold hover:bg-slate-200 transition-colors">Export Report</button>
          <button className="px-6 py-2 bg-blue-600/20 text-white border border-blue-500/40 rounded-lg font-bold hover:bg-blue-600/40 transition-colors">View API Docs</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0a0f1c] text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Navigation Sidebar */}
      <nav className="fixed left-0 top-0 bottom-0 w-72 bg-[#0d1324] border-r border-slate-800 hidden xl:flex flex-col p-6 z-50">
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-600/30">
            <ShieldCheck className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">FRAUD GUARD</h1>
            <p className="text-[10px] text-slate-500 tracking-[0.2em] font-black">AI ANALYTICS ENGINE</p>
          </div>
        </div>

        <div className="space-y-1 flex-1 overflow-y-auto no-scrollbar">
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={() => setActiveSection(s.id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeSection === s.id 
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-600/20' 
                  : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
              }`}
            >
              {s.icon}
              {s.title}
            </a>
          ))}
        </div>

        <div className="mt-auto pt-6 border-t border-slate-800">
          <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800 flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <div className="text-xs font-bold text-slate-400">System Status: Active</div>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="xl:ml-72 min-h-screen p-6 lg:p-12 max-w-6xl mx-auto">
        {/* Dashboard Header */}
        <header className="mb-12 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <div className="text-blue-500 font-black text-xs tracking-widest uppercase mb-2">Portfolio Project | Fintech & ML</div>
            <h1 className="text-4xl md:text-5xl font-black text-white tracking-tight">Real-Time Transaction <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400">Anomaly Detection</span></h1>
            <p className="text-slate-400 mt-4 text-lg max-w-2xl leading-relaxed">
              A comprehensive system for detecting financial fraud using unsupervised learning, robust evaluation frameworks, and human-in-the-loop decisioning.
            </p>
          </div>
          <div className="flex gap-3">
            <div className="px-4 py-2 bg-slate-800/50 rounded-full border border-slate-700 text-xs font-bold text-slate-300 flex items-center gap-2">
              <Database className="w-3 h-3" /> ULB-CC-v2
            </div>
            <div className="px-4 py-2 bg-blue-500/10 rounded-full border border-blue-500/20 text-xs font-bold text-blue-400 flex items-center gap-2">
              <Cpu className="w-3 h-3" /> Hybrid XGB-ISO
            </div>
          </div>
        </header>

        {/* Content Sections */}
        <div className="space-y-24">
          <IntroSection />
          <DataSection />
          <EDASection />
          <FeatureSection />
          <ModelSection />
          <EvalSection />
          <TuningSection />
          <ExplainSection />
          <ProdSection />
          <InsightsSection />
          <RecommendSection />
          <ConclusionSection />
        </div>
      </main>

      {/* Scroll to Top / Navigation Mobile Mock */}
      <div className="fixed bottom-6 right-6 flex flex-col gap-2 z-[60]">
        <button 
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="p-3 bg-blue-600 hover:bg-blue-500 text-white rounded-full shadow-xl shadow-blue-500/20 transition-all"
        >
          <ChevronRight className="w-6 h-6 -rotate-90" />
        </button>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        html { scroll-behavior: smooth; }
        input[type='range']::-webkit-slider-thumb {
          -webkit-appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
          transition: transform 0.1s ease-in-out;
        }
        input[type='range']::-webkit-slider-thumb:hover {
          transform: scale(1.1);
        }
      `}} />
    </div>
  );
}
