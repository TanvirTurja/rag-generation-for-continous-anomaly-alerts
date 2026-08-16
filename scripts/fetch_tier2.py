#!/usr/bin/env python3
"""
Tier-2 RAG corpus builder for the "RAG for continuous anomaly alerts" project.

Fetches OPEN-ACCESS, full-text biomedical articles from Europe PMC across six
topic buckets, converts each to clean Markdown, and writes:
  Tier2_literature/<bucket>/<PMCID>.md     (one per article)
  Tier2_literature/references.bib          (BibTeX for every article)
  Tier2_literature/manifest.csv            (metadata table)
  Tier2_literature/coverage_report.md      (per-bucket counts + gaps)

Free, no API key. All sources are CC0 / CC BY / CC BY-SA (Europe PMC OA subset).
"""
import urllib.request, urllib.parse, json, os, re, time, csv, html as ihtml

OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "Tier2_literature")
SEARCH   = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FT  = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
NCBI_FT  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={num}&rettype=xml&retmode=xml"
UA       = {"User-Agent": "rag-corpus-builder/1.0 (academic research)"}

# (folder, query, target). OPEN_ACCESS:Y appended automatically.
BUCKETS = [
    ("01_ppg_arrhythmia",
     '(photoplethysmography OR PPG) AND (arrhythmia OR atrial fibrillation) AND detection', 50),
    ("02_ppg_signal_quality",
     '(photoplethysmography OR PPG) AND ("signal quality" OR artifact OR "motion artifact")', 25),
    ("03_ecg_anomaly_ml",
     '(ECG OR electrocardiogram) AND ("anomaly detection" OR "arrhythmia classification") AND (deep learning OR machine learning)', 45),
    ("04_wearable_stress",
     '("stress detection" OR "affect detection") AND (wearable OR EDA OR electrodermal)', 35),
    ("05_continuous_monitoring",
     '("continuous monitoring" OR ambulatory) AND (wearable OR cardiac) AND anomaly', 25),
    ("06_biosignal_methods",
     '"anomaly detection" AND (biosignal OR physiological) AND (unsupervised OR "isolation forest")', 20),
]

# ---------------------------------------------------------------- helpers ----
def http_get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def search_page(query, pagesize, cursor):
    params = urllib.parse.urlencode({
        "query": query, "format": "json", "pageSize": pagesize,
        "resultType": "core", "cursorMark": cursor,
    })
    d = json.loads(http_get(SEARCH + "?" + params))
    return d.get("hitCount", 0), d.get("resultList", {}).get("result", []), d.get("nextCursorMark", "*")

def xml_to_text(xml):
    """Strip all JATS/XML tags, keep readable text. Robust to schema variants."""
    # drop elements that add noise
    xml = re.sub(r"<xref[^>]*>.*?</xref>", " ", xml, flags=re.S)
    xml = re.sub(r"<table-wrap[^>]*>.*?</table-wrap>", " ", xml, flags=re.S)
    xml = re.sub(r"<fig[^>]*>.*?</fig>", " ", xml, flags=re.S)
    # turn block boundaries into newlines
    xml = re.sub(r"</(p|sec|title|abstract|body|h1|h2|h3|list-item)>", "\n", xml, flags=re.S)
    xml = re.sub(r"<[^>]+>", " ", xml)
    xml = ihtml.unescape(xml)
    xml = re.sub(r"[ \t]+", " ", xml)
    xml = re.sub(r"\n\s*\n+", "\n\n", xml)
    return xml.strip()

def fetch_fulltext(pmcid):
    """Try Europe PMC fullTextXML, fall back to NCBI eutils. Return text or None."""
    num = pmcid.replace("PMC", "")
    for url in (EPMC_FT.format(pmcid=pmcid), NCBI_FT.format(num=num)):
        try:
            xml = http_get(url).decode("utf-8", "ignore")
            if "<body" in xml.lower() or "<article" in xml.lower():
                txt = xml_to_text(xml)
                if len(txt) > 1500:        # ignore stubs / abstract-only
                    return txt, ("EPMC" if "ebi.ac.uk" in url else "NCBI")
        except Exception:
            continue
    return None, None

def slugify(t, n=60):
    s = re.sub(r"[^A-Za-z0-9 ]+", "", t or "").strip().lower().split()
    return "_".join(s[:n])[:n] or "article"

# ----------------------------------------------------------------- main -----
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    bib_entries, manifest, stats = [], [], {}
    seen = set()   # global dedup by PMCID across buckets

    for folder, query, target in BUCKETS:
        bucket_dir = os.path.join(OUT_ROOT, folder)
        os.makedirs(bucket_dir, exist_ok=True)
        full_q = f"({query}) AND OPEN_ACCESS:Y"
        got, cursor, examined, failures = 0, "*", 0, 0
        print(f"\n=== {folder}  (target {target}) ===")

        while got < target:
            try:
                hitcount, results, nxt = search_page(full_q, 100, cursor)
            except Exception as e:
                print(f"  search error: {e}; retrying once")
                time.sleep(2); continue
            if not results:
                print(f"  no more results (total OA hits for bucket: {hitcount:,})")
                break
            for a in results:
                if got >= target:
                    break
                examined += 1
                pmcid = a.get("pmcid")
                if not pmcid or a.get("inEPMC") != "Y":
                    continue
                if pmcid in seen:
                    continue
                seen.add(pmcid)
                txt, source = fetch_fulltext(pmcid)
                time.sleep(0.2)   # be polite
                if not txt:
                    failures += 1
                    continue   # pmcid stays in `seen` so we don't retry the same failing id
                title  = (a.get("title") or "Untitled").strip().rstrip(".")
                auth   = a.get("authorString") or ""
                journal= a.get("journalTitle") or ""
                year   = a.get("pubYear") or ""
                doi    = a.get("doi") or ""
                lic     = a.get("license") or ""
                pmid    = a.get("pmid") or ""

                # markdown
                md = (f"# {title}\n\n"
                      f"**Authors:** {auth}  \n"
                      f"**Journal:** {journal} ({year})  \n"
                      f"**DOI:** {doi}  \n**PMCID:** {pmcid}  \n**PMID:** {pmid}  \n"
                      f"**License:** {lic}  \n**Source:** {source} (open access)\n\n"
                      f"---\n\n{txt}\n")
                fname = f"{pmcid}_{slugify(title)}.md"
                with open(os.path.join(bucket_dir, fname), "w", encoding="utf-8") as f:
                    f.write(md)

                # bibtex
                key = pmcid.replace("PMC", "PMC")
                bib = (f"@article{{{key},\n"
                       f"  title   = {{{title}}},\n"
                       f"  author  = {{{auth}}},\n"
                       f"  journal = {{{journal}}},\n"
                       f"  year    = {{{year}}},\n"
                       f"  doi     = {{{doi}}},\n"
                       f"  url     = {{https://europepmc.org/article/PMC/{pmcid.replace('PMC','')}}}\n}}\n\n")
                bib_entries.append(bib)
                manifest.append([pmcid, pmid, doi, folder, year, journal, lic,
                                 len(txt), fname])
                got += 1
            print(f"  examined={examined}  saved={got}/{target}  (failures={failures})")
            if nxt == cursor or nxt == "*":
                break
            cursor = nxt

        stats[folder] = (got, examined, failures)
        print(f"  -> {folder} done: {got} articles saved")

    # ---- write references.bib ----
    with open(os.path.join(OUT_ROOT, "references.bib"), "w", encoding="utf-8") as f:
        f.write("% Tier-2 RAG corpus — auto-generated from Europe PMC (Open Access)\n\n")
        f.writelines(bib_entries)

    # ---- write manifest.csv ----
    with open(os.path.join(OUT_ROOT, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmcid","pmid","doi","bucket","year","journal","license","text_chars","file"])
        w.writerows(manifest)

    # ---- coverage report ----
    total = sum(s[0] for s in stats.values())
    lines = ["# Tier-2 Corpus Coverage Report", "",
             f"**Total articles saved:** {total}", "",
             "| Bucket | Saved | Examined | Full-text failures |",
             "|---|---:|---:|---:|"]
    for folder, q, tgt in BUCKETS:
        g, ex, fl = stats[folder]
        lines.append(f"| {folder} | {g} (target {tgt}) | {ex} | {fl} |")
    lines += ["", f"**Output:** `{OUT_ROOT}`",
              "**Sources:** Europe PMC REST API (open access subset) + NCBI eutils fallback. "
              "All articles CC0 / CC BY / CC BY-SA."]
    with open(os.path.join(OUT_ROOT, "coverage_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n=== DONE: {total} articles. Report at {os.path.join(OUT_ROOT,'coverage_report.md')} ===")

if __name__ == "__main__":
    main()
