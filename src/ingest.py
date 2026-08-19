"""Download and clean financial filings for the local RAG corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.config import (
    DEFAULT_SEC_USER_AGENT,
    FILINGS_MANIFEST_PATH,
    RAW_DATA_DIR,
    load_environment,
)


DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "TSLA", "WMT"]
SEC_BASE_URL = "https://www.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{filename}"


@dataclass(frozen=True)
class FilingRecord:
    """Metadata for one cleaned filing in the local corpus."""

    ticker: str
    company: str
    fiscal_year: int | None
    form: str
    accession_number: str | None
    filing_date: str | None
    report_date: str | None
    source_url: str | None
    local_path: str


def sec_headers() -> dict[str, str]:
    """Return SEC-friendly request headers."""
    load_environment()
    return {
        "User-Agent": os.getenv("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
        "Accept-Encoding": "gzip, deflate",
        "Host": "",
    }


def client_get(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch a SEC URL with the correct Host header for that endpoint."""
    headers = sec_headers()
    headers["Host"] = httpx.URL(url).host or "www.sec.gov"
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return response


def load_ticker_map(client: httpx.Client) -> dict[str, dict[str, Any]]:
    """Load SEC's ticker-to-CIK mapping."""
    response = client_get(client, SEC_COMPANY_TICKERS_URL)
    raw_mapping = response.json()

    ticker_map: dict[str, dict[str, Any]] = {}
    for item in raw_mapping.values():
        ticker_map[item["ticker"].upper()] = {
            "cik": int(item["cik_str"]),
            "company": item["title"],
        }
    return ticker_map


def extract_filings(filings_payload: dict[str, list[str]]) -> list[dict[str, str]]:
    """Convert SEC's column-oriented filing payload into row dictionaries."""
    filings = [
        {
            "accession_number": accession,
            "filing_date": filing_date,
            "report_date": report_date,
            "form": form,
            "primary_document": primary_doc,
        }
        for accession, filing_date, report_date, form, primary_doc in zip(
            filings_payload["accessionNumber"],
            filings_payload["filingDate"],
            filings_payload["reportDate"],
            filings_payload["form"],
            filings_payload["primaryDocument"],
            strict=True,
        )
    ]
    return filings


def has_matching_annual_filing(
    filings: list[dict[str, str]],
    fiscal_year: int | None,
    form_type: str,
) -> bool:
    """Return whether loaded filing rows already contain the requested annual filing."""
    if fiscal_year is None:
        return any(filing["form"] == form_type for filing in filings)
    return any(
        filing["form"] == form_type
        and filing["report_date"].startswith(str(fiscal_year))
        for filing in filings
    )


def load_company_filings(
    client: httpx.Client,
    cik: int,
    fiscal_year: int | None,
    form_type: str,
) -> list[dict[str, str]]:
    """Load SEC filing rows, stopping once the requested annual filing is present."""
    submissions = client_get(client, SEC_SUBMISSIONS_URL.format(cik=cik)).json()
    filings = extract_filings(submissions["filings"]["recent"])
    if has_matching_annual_filing(filings, fiscal_year, form_type):
        return filings

    for file_info in submissions["filings"].get("files", []):
        archive_url = SEC_SUBMISSIONS_FILE_URL.format(filename=file_info["name"])
        archive_payload = client_get(client, archive_url).json()
        filings.extend(extract_filings(archive_payload))
        if has_matching_annual_filing(filings, fiscal_year, form_type):
            return filings

    return filings


def find_annual_filing(
    filings: list[dict[str, str]],
    fiscal_year: int | None,
    form_type: str,
) -> dict[str, str]:
    """Find the most relevant 10-K-style filing from SEC filing rows."""
    annual_filings = [filing for filing in filings if filing["form"] == form_type]

    if fiscal_year is not None:
        matching_year = [
            filing
            for filing in annual_filings
            if filing["report_date"].startswith(str(fiscal_year))
        ]
        if matching_year:
            return matching_year[0]
        raise RuntimeError(f"No {form_type} filing found for report year {fiscal_year}.")

    if not annual_filings:
        raise RuntimeError(f"No {form_type} filings found in SEC submissions.")

    return annual_filings[0]


def filing_document_url(cik: int, accession_number: str, primary_document: str) -> str:
    """Build the SEC archive URL for a filing's primary document."""
    cik_path = str(cik)
    accession_path = accession_number.replace("-", "")
    return (
        f"{SEC_BASE_URL}/Archives/edgar/data/"
        f"{cik_path}/{accession_path}/{primary_document}"
    )


def clean_filing_text(raw_content: str) -> str:
    """Convert HTML or SEC text content into normalized plain text."""
    soup = BeautifulSoup(raw_content, "html.parser")
    for tag in soup(["script", "style", "noscript", "ix:header"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip() + "\n"


def write_clean_filing(
    *,
    ticker: str,
    company: str,
    fiscal_year: int | None,
    form: str,
    accession_number: str | None,
    filing_date: str | None,
    report_date: str | None,
    source_url: str | None,
    raw_content: str,
) -> FilingRecord:
    """Clean one filing and write it to data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    year_label = fiscal_year or (report_date[:4] if report_date else "unknown")
    output_path = RAW_DATA_DIR / f"{ticker.upper()}_{year_label}.txt"
    output_path.write_text(clean_filing_text(raw_content), encoding="utf-8")

    return FilingRecord(
        ticker=ticker.upper(),
        company=company,
        fiscal_year=fiscal_year,
        form=form,
        accession_number=accession_number,
        filing_date=filing_date,
        report_date=report_date,
        source_url=source_url,
        local_path=str(output_path.relative_to(RAW_DATA_DIR.parent.parent)),
    )


def download_sec_filings(
    tickers: list[str],
    fiscal_year: int | None,
    form_type: str,
) -> list[FilingRecord]:
    """Download and clean annual filings from SEC EDGAR."""
    records: list[FilingRecord] = []
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        ticker_map = load_ticker_map(client)

        for ticker in tickers:
            normalized_ticker = ticker.upper()
            if normalized_ticker not in ticker_map:
                raise RuntimeError(f"Ticker not found in SEC mapping: {ticker}")

            cik = ticker_map[normalized_ticker]["cik"]
            company = ticker_map[normalized_ticker]["company"]
            filings = load_company_filings(client, cik, fiscal_year, form_type)
            filing = find_annual_filing(filings, fiscal_year, form_type)
            source_url = filing_document_url(
                cik,
                filing["accession_number"],
                filing["primary_document"],
            )

            response = client_get(client, source_url)
            record = write_clean_filing(
                ticker=normalized_ticker,
                company=company,
                fiscal_year=fiscal_year,
                form=filing["form"],
                accession_number=filing["accession_number"],
                filing_date=filing["filing_date"],
                report_date=filing["report_date"],
                source_url=source_url,
                raw_content=response.text,
            )
            records.append(record)
            print(f"Wrote {record.local_path}")

    return records


def infer_local_metadata(path: Path) -> tuple[str, int | None]:
    """Infer ticker and year from a local filename like AAPL_2023.html."""
    match = re.search(r"([A-Za-z]{1,6})[_-](20\d{2})", path.stem)
    if not match:
        return path.stem.upper(), None
    return match.group(1).upper(), int(match.group(2))


def clean_local_filings(input_dir: Path) -> list[FilingRecord]:
    """Clean manually downloaded filing files from a local folder."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Manual input folder does not exist: {input_dir}")

    records: list[FilingRecord] = []
    candidates = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in {".txt", ".html", ".htm"}
    )
    if not candidates:
        raise RuntimeError(f"No .txt, .html, or .htm files found in {input_dir}")

    for path in candidates:
        ticker, fiscal_year = infer_local_metadata(path)
        raw_content = path.read_text(encoding="utf-8", errors="ignore")
        record = write_clean_filing(
            ticker=ticker,
            company=ticker,
            fiscal_year=fiscal_year,
            form="10-K",
            accession_number=None,
            filing_date=None,
            report_date=str(fiscal_year) if fiscal_year else None,
            source_url=str(path),
            raw_content=raw_content,
        )
        records.append(record)
        print(f"Wrote {record.local_path}")

    return records


def write_manifest(records: list[FilingRecord]) -> None:
    """Write filing metadata for downstream chunking and documentation."""
    FILINGS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    FILINGS_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FILINGS_MANIFEST_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or clean financial filings into data/raw/."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to download from SEC EDGAR.",
    )
    parser.add_argument(
        "--fiscal-year",
        type=int,
        default=2023,
        help="Fiscal year to select by report date. Use 0 for latest available.",
    )
    parser.add_argument(
        "--form-type",
        default="10-K",
        help="SEC annual filing form to download, usually 10-K.",
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        help="Clean local .txt/.html filings instead of downloading from SEC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fiscal_year = None if args.fiscal_year == 0 else args.fiscal_year

    if args.manual_dir:
        records = clean_local_filings(args.manual_dir)
    else:
        records = download_sec_filings(args.tickers, fiscal_year, args.form_type)

    write_manifest(records)


if __name__ == "__main__":
    main()
