"""Public FFIEC bulk-form probe and narrow-period acquisition.

Uses the site's public HTML form and session cookies, never a private PWS endpoint.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from treasuryshield.orchestration.state import RunState

URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
PRODUCT = "ReportingSeriesSinglePeriod"


def _form(soup: BeautifulSoup) -> dict[str, str]:
    data: dict[str, str] = {}
    for item in soup.select("input[name]"):
        if item.get("type", "").lower() == "hidden":
            data[str(item["name"])] = str(item.get("value", ""))
    return data


def probe(root: Path) -> dict[str, object]:
    session = requests.Session()
    session.headers["User-Agent"] = "TreasuryShield-MRM public-data research/0.1"
    response = session.get(URL, timeout=45)
    response.raise_for_status()
    if "login" in response.url.lower():
        raise PermissionError(f"FFIEC redirected to login: {response.url}")
    soup = BeautifulSoup(response.text, "html.parser")
    data = _form(soup)
    data["ctl00$MainContentHolder$ListBox1"] = PRODUCT
    data["__EVENTTARGET"] = "ctl00$MainContentHolder$ListBox1"
    data["__EVENTARGUMENT"] = ""
    response2 = session.post(URL, data=data, timeout=60)
    response2.raise_for_status()
    soup2 = BeautifulSoup(response2.text, "html.parser")
    choices = [
        {"value": str(option.get("value", "")), "text": option.get_text(" ", strip=True)}
        for option in soup2.select("#DatesDropDownList option")
    ]
    result: dict[str, object] = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "url": response2.url,
        "status": response2.status_code,
        "selected_product": PRODUCT,
        "period_choices": choices,
        "login_detected": "login" in response2.url.lower() or "sign in" in response2.text.lower(),
        "response_bytes": len(response2.content),
    }
    out = root / "data" / "manifests" / "ffiec_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:6000])
    return result


def download_period(root: Path, period: str) -> Path:
    """Download a single public Call Report quarter in tab-delimited ZIP format."""
    session = requests.Session()
    session.headers["User-Agent"] = "TreasuryShield-MRM public-data research/0.1"
    response = session.get(URL, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    data = _form(soup)
    data.update({
        "ctl00$MainContentHolder$ListBox1": PRODUCT,
        "__EVENTTARGET": "ctl00$MainContentHolder$ListBox1",
        "__EVENTARGUMENT": "",
    })
    response = session.post(URL, data=data, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    choices = {option.get_text(" ", strip=True): str(option.get("value", ""))
               for option in soup.select("#DatesDropDownList option")}
    if period not in choices:
        raise ValueError(f"FFIEC reporting period unavailable: {period}")
    data = _form(soup)
    data.update({
        "ctl00$MainContentHolder$ListBox1": PRODUCT,
        "ctl00$MainContentHolder$DatesDropDownList": choices[period],
        "ctl00$MainContentHolder$FormatType": "TSVRadioButton",
        "ctl00$MainContentHolder$TabStrip1$Download_0": "Download",
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
    })
    response = session.post(URL, data=data, timeout=300)
    response.raise_for_status()
    if not response.content.startswith(b"PK\x03\x04"):
        detail = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
        raise RuntimeError(f"FFIEC did not return a ZIP for {period}: {detail[:600]}")
    out = root / "data" / "raw" / "ffiec" / f"call_{period.replace('/', '')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Raw file already present: {out}")
    out.write_bytes(response.content)
    print(f"Downloaded {period}: {out.stat().st_size} bytes -> {out}")
    return out


def download_range(root: Path, start_year: int = 2017, end_period: str = "06/30/2026") -> list[Path]:
    """Resume a bounded 051-era Enhanced Panel acquisition."""
    session = requests.Session()
    session.headers["User-Agent"] = "TreasuryShield-MRM public-data research/0.1"
    response = session.get(URL, timeout=45)
    response.raise_for_status()
    data = _form(BeautifulSoup(response.text, "html.parser"))
    data.update({
        "ctl00$MainContentHolder$ListBox1": PRODUCT,
        "__EVENTTARGET": "ctl00$MainContentHolder$ListBox1",
        "__EVENTARGUMENT": "",
    })
    response = session.post(URL, data=data, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    choices = {
        option.get_text(" ", strip=True): str(option.get("value", ""))
        for option in soup.select("#DatesDropDownList option")
    }
    end = datetime.strptime(end_period, "%m/%d/%Y")
    selected = sorted(
        ((datetime.strptime(text, "%m/%d/%Y"), text, value)
         for text, value in choices.items()
         if datetime.strptime(text, "%m/%d/%Y").year >= start_year
         and datetime.strptime(text, "%m/%d/%Y") <= end),
        reverse=True,
    )
    if not selected:
        raise ValueError("No FFIEC periods selected")
    base_form = _form(soup)
    raw = root / "data" / "raw" / "ffiec"
    raw.mkdir(parents=True, exist_ok=True)
    outputs = [str((raw / f"call_{text.replace('/', '')}.zip").relative_to(root)).replace("\\", "/")
               for _, text, _ in selected]

    def acquire() -> list[Path]:
        result = []
        for _, text, value in selected:
            path = raw / f"call_{text.replace('/', '')}.zip"
            if path.exists():
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"Existing FFIEC archive failed CRC: {path}")
                print(f"FFIEC cached {text}: {path.stat().st_size} bytes", flush=True)
                result.append(path)
                continue
            form = dict(base_form)
            form.update({
                "ctl00$MainContentHolder$ListBox1": PRODUCT,
                "ctl00$MainContentHolder$DatesDropDownList": value,
                "ctl00$MainContentHolder$FormatType": "TSVRadioButton",
                "ctl00$MainContentHolder$TabStrip1$Download_0": "Download",
                "__EVENTTARGET": "", "__EVENTARGUMENT": "",
            })
            reply = session.post(URL, data=form, timeout=300)
            reply.raise_for_status()
            if not reply.content.startswith(b"PK\x03\x04"):
                detail = BeautifulSoup(reply.text, "html.parser").get_text(" ", strip=True)
                raise RuntimeError(f"FFIEC did not return ZIP for {text}: {detail[:500]}")
            path.write_bytes(reply.content)
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Downloaded FFIEC archive failed CRC: {path}")
            print(f"FFIEC acquired {text}: {path.stat().st_size} bytes", flush=True)
            result.append(path)
            time.sleep(0.2)
        return result

    state = RunState(root)
    state.run(
        "ffiec_bulk_2017_2026q2", "ACQUIRING_DATA",
        {"url": URL, "product": PRODUCT, "start_year": start_year, "end_period": end_period,
         "periods": [text for _, text, _ in selected]},
        outputs, acquire, resume=True,
    )
    return [root / name for name in outputs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-period", help="MM/DD/YYYY reporting period")
    parser.add_argument("--download-range-start", type=int, help="Starting calendar year for all public quarters")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.download_range_start:
        download_range(root, args.download_range_start)
    elif args.download_period:
        download_period(root, args.download_period)
    else:
        probe(root)
