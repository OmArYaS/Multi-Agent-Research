import ReactMarkdown from 'react-markdown'

export default function ReportView({ report }) {
  if (!report) {
    return (
      <div className="panel report-panel">
        <h2>Final report</h2>
        <p className="muted">No report was produced for this run.</p>
      </div>
    )
  }

  return (
    <div className="panel report-panel">
      <h2>Final report</h2>
      <article className="report">
        <h1>{report.title}</h1>
        {report.executive_summary && (
          <section>
            <h3>Executive summary</h3>
            <p>{report.executive_summary}</p>
          </section>
        )}
        {report.key_takeaways?.length > 0 && (
          <section>
            <h3>Key takeaways</h3>
            <ul>
              {report.key_takeaways.map((t, i) => <li key={i}>{t}</li>)}
            </ul>
          </section>
        )}
        <section>
          <ReactMarkdown
            components={{
              a: ({ node, ...props }) => (
                <a {...props} target="_blank" rel="noopener noreferrer" />
              ),
            }}
          >
            {report.body || ''}
          </ReactMarkdown>
        </section>
        {report.sources?.length > 0 && (
          <section>
            <h3>Sources</h3>
            <ol className="sources">
              {report.sources.map((source, i) => (
                <li key={i}>
                  {source.url ? (
                    <a href={source.url} target="_blank" rel="noopener noreferrer">
                      {source.title || source.url}
                    </a>
                  ) : (
                    source.title
                  )}
                  {source.snippet && <span className="snippet"> — {source.snippet}</span>}
                </li>
              ))}
            </ol>
          </section>
        )}
      </article>
    </div>
  )
}
