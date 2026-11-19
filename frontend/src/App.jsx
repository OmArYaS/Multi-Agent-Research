import { useCallback, useEffect, useRef, useState } from 'react'
import RunTimeline from './components/RunTimeline.jsx'
import ReportView from './components/ReportView.jsx'

const API = {
  async models() {
    const res = await fetch('/api/models')
    if (!res.ok) throw new Error(`Failed to fetch models (${res.status})`)
    return (await res.json()).models || []
  },
  async startRun(payload) {
    const res = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error(detail.detail || `Failed to start run (${res.status})`)
    }
    return res.json()
  },
  async getRun(runId) {
    const res = await fetch(`/api/runs/${runId}`)
    if (!res.ok) throw new Error(`Failed to fetch run (${res.status})`)
    return res.json()
  },
}

export default function App() {
  const [topic, setTopic] = useState('')
  const [model, setModel] = useState('')
  const [models, setModels] = useState([])
  const [maxIterations, setMaxIterations] = useState(2)
  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const activeRunRef = useRef(0)

  useEffect(() => {
    API.models()
      .then((list) => {
        setModels(list)
        if (list.length && !model) setModel(list[0])
      })
      .catch(() => setModels([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = useCallback(async (event) => {
    event.preventDefault()
    if (!topic.trim()) {
      setError('Please enter a research topic.')
      return
    }
    const token = ++activeRunRef.current
    setLoading(true)
    setError('')
    setRun(null)
    try {
      const payload = {
        topic: topic.trim(),
        model: model.trim() || undefined,
        max_iterations: Number(maxIterations),
      }
      const started = await API.startRun(payload)
      let result = await API.getRun(started.run_id)
      // Defensive polling in case the API later becomes asynchronous.
      let attempts = 0
      while (result.status === 'running' && attempts < 30 && token === activeRunRef.current) {
        await new Promise((r) => setTimeout(r, 700))
        result = await API.getRun(started.run_id)
        attempts += 1
      }
      if (token === activeRunRef.current) setRun(result)
    } catch (err) {
      if (token === activeRunRef.current) setError(err.message || 'Unexpected error')
    } finally {
      if (token === activeRunRef.current) setLoading(false)
    }
  }, [topic, model, maxIterations])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Multi-Agent Research Orchestrator</h1>
          <p>Planner &rarr; Researcher &rarr; Analyst &rarr; Reviewer &rarr; Editor, powered by local Ollama.</p>
        </div>
        <span className={`connection-badge ${models.length ? 'online' : 'offline'}`}>
          {models.length ? `${models.length} local model${models.length > 1 ? 's' : ''}` : 'Ollama offline'}
        </span>
      </header>

      <main className="layout">
        <aside className="panel controls">
          <h2>New research run</h2>
          <form onSubmit={start}>
            <label htmlFor="topic">Research topic</label>
            <textarea
              id="topic"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. Impact of small language models on edge devices"
              rows={4}
              disabled={loading}
            />

            <label htmlFor="model">Ollama model</label>
            <input
              id="model"
              list="model-options"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="qwen2.5:7b-instruct-q4_k_m"
              disabled={loading}
            />
            <datalist id="model-options">
              {models.map((m) => <option key={m} value={m} />)}
            </datalist>
            <small className="hint">Any pulled Ollama model works: gemma2, deepseek-r1, llama3.1, ...</small>

            <label htmlFor="iterations">Max revision iterations</label>
            <input
              id="iterations"
              type="number"
              min={1}
              max={5}
              value={maxIterations}
              onChange={(e) => setMaxIterations(e.target.value)}
              disabled={loading}
            />
            <small className="hint">Caps the review &rarr; research loop (default 2).</small>

            <button type="submit" disabled={loading}>
              {loading ? 'Running agents...' : 'Start research'}
            </button>
          </form>
          {error && <div className="error-card">{error}</div>}
          {loading && (
            <div className="loading-note">
              <span className="spinner" /> Agents are researching and critiquing. This can take a minute.
            </div>
          )}
        </aside>

        <section className="results">
          {!run && !loading && (
            <div className="empty-state">
              <h2>Ready when you are</h2>
              <p>Enter a topic on the left and start the workflow. Agent steps, reviewer feedback, and the final Markdown report will appear here.</p>
            </div>
          )}
          {run && (
            <div className="results-grid">
              <RunTimeline steps={run.steps || []} review={run.review} error={run.error} />
              <ReportView report={run.report} />
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
