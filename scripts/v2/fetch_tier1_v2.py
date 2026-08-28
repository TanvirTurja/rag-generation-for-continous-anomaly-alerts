"""
fetch_tier1_v2.py — Expand Tier-1 with wearable-relevant guidance (fix/v2 W3).

v1's Tier-1 (syncope + VA/SCD guidelines) matched none of the alerts the system
actually produces (393/398 alerts cited zero guideline content). This fetches
guidance aligned with the real alert space (wearables, PPG, AF screening,
arrhythmia monitoring):

  1. Svennberg et al. 2022, "How to use digital devices to detect and manage
     arrhythmias: an EHRA practical guide", EP Europace 24(6):979-1005.
     DOI 10.1093/europace/euac038. Free-to-read on PMC (PMC11636571).
  2. Joglar et al. 2024, "2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and
     Management of Atrial Fibrillation" (JACC version), J Am Coll Cardiol
     83(1):109-279. DOI 10.1016/j.jacc.2023.08.017. Free-to-read on PMC
     (PMC11104284).
  3. Van Gelder et al. 2024, "2024 ESC Guidelines for the management of atrial
     fibrillation", Eur Heart J 45(36):3314-3414. DOI 10.1093/eurheartj/ehae178.
     OUP blocks programmatic access (403); attempted via escardio.org mirror;
     recorded as excluded-with-reason if unreachable.

Provenance strategy: Europe PMC fullTextXML -> NCBI efetch -> PMC article HTML
(free-to-read pages) -> publisher PDF. No paywalled content enters the corpus;
every document's access route and license status is recorded in manifest.csv.

Outputs: Dataset/Tier1_v2/<slug>.txt + Dataset/Tier1_v2/manifest.csv
"""

import csv
import html as ihtml
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Dataset" / "Tier1_v2"
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={num}&rettype=xml&retmode=xml"

DOCS = [
    {"slug": "ehra2022_digital_devices_arrhythmias",
     "title": "How to use digital devices to detect and manage arrhythmias: an EHRA practical guide",
     "doi": "10.1093/europace/euac038", "pmcid": "PMC11636571",
     "html_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11636571/",
     "pdf_urls": []},
    {"slug": "accaha2023_af_guideline",
     "title": "2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation",
     "doi": "10.1016/j.jacc.2023.08.017", "pmcid": "PMC11104284",
     "html_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11104284/",
     "pdf_urls": []},
    {"slug": "esc2024_af_guideline",
     "title": "2024 ESC Guidelines for the management of atrial fibrillation",
     "doi": "10.1093/eurheartj/ehae178", "pmcid": None, "html_url": None,
     "landing_url": "https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/Atrial-Fibrillation-Guidelines",
     "pdf_urls": []},
]


def http_get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def xml_to_text(xml):
    xml = re.sub(r"<xref[^>]*>.*?</xref>", " ", xml, flags=re.S)
    xml = re.sub(r"<table-wrap[^>]*>.*?</table-wrap>", " ", xml, flags=re.S)
    xml = re.sub(r"<fig[^>]*>.*?</fig>", " ", xml, flags=re.S)
    xml = re.sub(r"</(p|sec|title|abstract|body|h1|h2|h3|list-item)>", "\n", xml, flags=re.S)
    xml = re.sub(r"<[^>]+>", " ", xml)
    xml = ihtml.unescape(xml)
    xml = re.sub(r"[ \t]+", " ", xml)
    xml = re.sub(r"\n\s*\n+", "\n\n", xml)
    return xml.strip()


def html_to_text(html):
    # strip scripts/styles, then tags; keep block structure
    txt = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"</(p|div|section|h1|h2|h3|h4|li|tr|br)>", "\n", txt, flags=re.I)
    txt = re.sub(r"<br\s*/?>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = ihtml.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


def try_fetch(doc):
    attempts = []
    pmcid = doc.get("pmcid")
    if pmcid:
        for name, url in [("EPMC-XML", EPMC_FT.format(pmcid=pmcid)),
                          ("NCBI-efetch", EFETCH.format(num=pmcid.replace("PMC", "")))]:
            try:
                xml = http_get(url).decode("utf-8", "ignore")
                if "<body" in xml.lower() and "</body>" in xml.lower():
                    txt = xml_to_text(xml)
                    if len(txt) > 20000:  # guidelines are long; stubs are ~5k
                        return txt, f"{name} {pmcid}", "open-access full text (API)"
                attempts.append(f"{name}: stub/short")
            except Exception as e:
                attempts.append(f"{name}: {type(e).__name__}")
        if doc.get("html_url"):
            try:
                html = http_get(doc["html_url"]).decode("utf-8", "ignore")
                # PMC HTML: keep only the article body section if isolatable
                m = re.search(r'(?s)<section[^>]*id="body"[^>]*>.*?(?=<section[^>]*id="(ack|ref)|<footer)',
                              html)
                txt = html_to_text(m.group(0) if m else html)
                if len(txt) > 20000:
                    return txt, f"PMC HTML {doc['html_url']}", "free-to-read (non-OA deposit)"
                attempts.append(f"PMC HTML: too short ({len(txt)} chars)")
            except Exception as e:
                attempts.append(f"PMC HTML: {type(e).__name__} {e}")
    if doc.get("landing_url"):
        try:
            html = http_get(doc["landing_url"]).decode("utf-8", "ignore")
            links = re.findall(r'href="([^"]+\.pdf[^"]*)"', html)
            for link in links[:10]:
                if link.startswith("/"):
                    link = "https://www.escardio.org" + link
                try:
                    data = http_get(link)
                    if data[:4] == b"%PDF":
                        import tempfile
                        import fitz
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                            f.write(data)
                            tmp = f.name
                        d = fitz.open(tmp)
                        txt = "\n".join(page.get_text() for page in d)
                        d.close()
                        if len(txt) > 20000:
                            return txt, f"web PDF {link}", "free PDF mirror"
                except Exception:
                    continue
            attempts.append(f"landing page: {len(links)} pdf links, none usable")
        except Exception as e:
            attempts.append(f"landing page: {type(e).__name__}")
    for url in doc["pdf_urls"]:
        try:
            data = http_get(url)
            if data[:4] != b"%PDF":
                attempts.append(f"pdf {url[-40:]}: not a PDF")
                continue
            import tempfile
            import fitz
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(data)
                tmp = f.name
            d = fitz.open(tmp)
            txt = "\n".join(page.get_text() for page in d)
            d.close()
            if len(txt) > 5000:
                return txt, f"web PDF {url}", "free PDF mirror"
            attempts.append(f"pdf {url[-40:]}: too short")
        except Exception as e:
            attempts.append(f"pdf {url[-40:]}: {type(e).__name__}")
    return None, "; ".join(attempts), "unknown"


def main():
    rows = []
    for doc in DOCS:
        print(f"fetching: {doc['title'][:60]}...", flush=True)
        text, prov, lic = try_fetch(doc)
        if text is None:
            print(f"  EXCLUDED: {prov}", flush=True)
            rows.append({"slug": doc["slug"], "title": doc["title"], "doi": doc["doi"],
                         "status": "excluded", "provenance": prov, "license": lic, "words": 0})
            continue
        words = len(text.split())
        (OUT / f"{doc['slug']}.txt").write_text(text, encoding="utf-8")
        print(f"  OK: {words:,} words via {prov} ({lic})", flush=True)
        rows.append({"slug": doc["slug"], "title": doc["title"], "doi": doc["doi"],
                     "status": "included", "provenance": prov, "license": lic, "words": words})
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
