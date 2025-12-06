export default function Home() {
  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <header className="border-b border-slate-700 bg-slate-800/50 backdrop-blur">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">MathLedger</h1>
          <div className="text-sm text-slate-400">
            Google Maps for Math & Truth
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Universe View</h2>
          <p className="text-slate-400">
            Navigate the proof graph • Explore theorem dependencies • Verify claims
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Graph Canvas Placeholder */}
          <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-8 min-h-[500px] flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl mb-4">🗺️</div>
              <h3 className="text-xl font-semibold mb-2">Graph Canvas</h3>
              <p className="text-slate-400 max-w-md">
                Zoomable graph visualization will render here
              </p>
              <div className="mt-6 flex gap-3 justify-center">
                <div className="px-3 py-1 bg-green-500/20 text-green-400 rounded border border-green-500/50 text-sm">
                  PROVED
                </div>
                <div className="px-3 py-1 bg-slate-500/20 text-slate-400 rounded border border-slate-500/50 text-sm">
                  PENDING
                </div>
                <div className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded border border-yellow-500/50 text-sm">
                  ABSTAIN
                </div>
              </div>
            </div>
          </div>

          {/* Factory Panel */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <span className="text-2xl">⚙️</span>
              Factory Floor
            </h3>
            <div className="space-y-3">
              {['Cursor A', 'Replit A', 'Grok A', 'Gemini A'].map((agent) => (
                <div
                  key={agent}
                  className="flex items-center justify-between p-3 bg-slate-900/50 rounded border border-slate-700/50"
                >
                  <span className="text-sm font-medium">{agent}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                    <span className="text-xs text-slate-400">ACTIVE</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sample Theorems */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <TheoremCard
            id="T1"
            label="Modus Ponens"
            status="PROVED"
            deps={0}
          />
          <TheoremCard
            id="T2"
            label="Export Linter Soundness"
            status="PENDING"
            deps={3}
          />
        </div>
      </main>

      <footer className="mt-12 border-t border-slate-700 bg-slate-800/50 py-4">
        <div className="container mx-auto px-4 text-center text-sm text-slate-400">
          Connected to <span className="text-slate-300 font-mono">mathledger.ai</span>
          {" • "}
          <span className="text-green-400">UI ✓</span>
          {" • "}
          <span className="text-slate-500">API —</span>
          {" • "}
          <span className="text-slate-500">Bridge —</span>
        </div>
      </footer>
    </div>
  );
}

function TheoremCard({
  id,
  label,
  status,
  deps,
}: {
  id: string;
  label: string;
  status: string;
  deps: number;
}) {
  const statusColors = {
    PROVED: 'bg-green-500/20 text-green-400 border-green-500/50',
    PENDING: 'bg-slate-500/20 text-slate-400 border-slate-500/50',
    ABSTAIN: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 hover:border-slate-600 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-xs text-slate-500 font-mono">{id}</div>
          <h4 className="font-semibold">{label}</h4>
        </div>
        <div
          className={`px-2 py-1 text-xs rounded border ${
            statusColors[status as keyof typeof statusColors]
          }`}
        >
          {status}
        </div>
      </div>
      <div className="flex items-center gap-4 text-xs text-slate-400 mt-3">
        <span>{deps} dependencies</span>
        {status === 'PENDING' && (
          <button className="ml-auto px-3 py-1 bg-blue-500/20 text-blue-400 rounded border border-blue-500/50 hover:bg-blue-500/30 transition-colors">
            Verify with POA
          </button>
        )}
      </div>
    </div>
  );
}
