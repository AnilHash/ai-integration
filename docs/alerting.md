### Alert Matrix — ai-infra-portfolio

| Signal               | Category  | Type                 | Where It Runs    | Threshold                    | Severity                        | Disposition                                                      |
| -------------------- | --------- | -------------------- | ---------------- | ---------------------------- | ------------------------------- | ---------------------------------------------------------------- |
| CI eval gate         | Pre-merge | Absolute floor       | GitHub Actions   | pass rate < 83%              | Blocks merge                    | Never a page — blocks shipping instead                           |
| Error rate           | Runtime   | Threshold (absolute) | Local script     | > 5% of last 20 requests     | Critical                        | 3am page — system is actually broken                             |
| Request volume       | Runtime   | Threshold (floor)    | Langfuse Monitor | count < 1 / hour             | Critical                        | 3am page — possible total outage                                 |
| P95 latency          | Runtime   | Threshold            | Langfuse Monitor | > 30s, sustained 1hr         | Warning                         | Next business day — slow, not down                               |
| Daily cost           | Runtime   | Threshold            | Langfuse Monitor | > budget (symbolic today)    | Warning                         | Next business day — becomes real post-Pillar 4/12                |
| Output quality drift | Runtime   | Anomaly (z-test)     | Local script     | p < 0.05, sustained 2 checks | Warning → Critical if sustained | Same-day investigate; escalate if persistent                     |
| Input (data) drift   | Runtime   | Anomaly (z-test)     | Local script     | p < 0.05                     | Informational                   | Dashboard/log only — informs golden set updates, not an incident |
