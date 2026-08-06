"""
Checks the LOCAL /admin/drift-report and posts a Slack alert if anything is flagged. Talks to localhost because it runs on the same machine as the app - this is NOT a substitute for cloud-based always-on monitoring.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
API_BASE = "http://localhost:8000"


def check_and_alert():
    resp = requests.get(
        f"{API_BASE}/admin/drift-report", params={"baseline_n": 50, "current_n": 20}
    )
    resp.raise_for_status()
    report = resp.json()

    if "error" in report:
        print(report["error"])
        return

    flags = []
    if report["reliability"]["flagged"]:
        flags.append(
            f"Error rate: {report['reliability']['current_errror_rate']:.1%} "
            f"(baseline {report['reliability']['baseline_error_rate']:.1%})"
        )

    if report["output_quality_drift"]["flagged"]:
        flags.append(
            f"Output quality drift: citation rate"
            f"{report['output_quality_drift']['current_cite_rate']:.1%} vs "
            f"baseline {report['output_quality_drift']['baseline_cite_rate']:.1%}"
        )

    if report["input_drift"]["flagged"]:
        flags.append(
            f"Input drift: mean query length "
            f"{report['input_drift']['current_mean_length']} vs "
            f"baseline {report['input_drift']['baseline_mean_length']}"
        )
    if not flags:
        print("No drift flagged")
        return

    message = "Drift Alert - ai-infra-portfolio*\n" + "\n".join(flags)
    print(message)

    if SLACK_WEBHOOK_URL:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message})
    else:
        print("(SLACK_WEBHOOK_URL not set - printed only, not sent)")


if __name__ == "__main__":
    check_and_alert()
