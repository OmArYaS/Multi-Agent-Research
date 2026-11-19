const AGENT_COLORS = {
  Planner: '#6366f1',
  Researcher: '#0ea5e9',
  Analyst: '#8b5cf6',
  Reviewer: '#f59e0b',
  Editor: '#10b981',
}

function VerdictBadge({ verdict }) {
  const isPass = verdict === 'pass'
  return (
    <span className={`verdict ${isPass ? 'pass' : 'revise'}`}>
      {isPass ? 'PASS' : 'REVISE'}
    </span>
  )
}

export default function RunTimeline({ steps, review, error }) {
  return (
    <div className="panel timeline-panel">
      <h2>Agent execution</h2>
      {error && <div className="error-card">{error}</div>}
      <ol className="timeline">
        {steps.map((step, index) => (
          <li key={index} className="timeline-item">
            <span
              className="agent-dot"
              style={{ backgroundColor: AGENT_COLORS[step.agent] || '#94a3b8' }}
            >
              {index + 1}
            </span>
            <div className="timeline-body">
              <div className="timeline-heading">
                <strong>{step.agent}</strong>
                <span className={`status status-${step.status}`}>{step.status}</span>
              </div>
              <div className="step-name">{step.step}</div>
              {step.detail && <p className="step-detail">{step.detail}</p>}
              <time>{step.timestamp ? new Date(step.timestamp).toLocaleString() : ''}</time>
            </div>
          </li>
        ))}
      </ol>

      {review && (
        <div className="review-card">
          <div className="review-heading">
            <h3>Reviewer feedback</h3>
            <VerdictBadge verdict={review.verdict} />
          </div>
          <div className="score">Quality score: <strong>{review.score}/100</strong></div>
          <p>{review.feedback}</p>
          {review.required_changes?.length > 0 && (
            <>
              <h4>Required changes</h4>
              <ul>{review.required_changes.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </>
          )}
          {review.missing_elements?.length > 0 && (
            <>
              <h4>Missing elements</h4>
              <ul>{review.missing_elements.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </>
          )}
        </div>
      )}
    </div>
  )
}
