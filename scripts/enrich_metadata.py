#!/usr/bin/env python3
"""
Enrich Tier-2 corpus metadata in place.

Fixes the journal field (was empty: it is nested under journalInfo.journal)
and fills missing authors/DOI by re-querying Europe PMC per PMCID.
Patches: references.bib, manifest.csv, and the **Journal:**/**Authors:**/**DOI:**
header lines inside each markdown file. Does NOT re-download full text.
"""
import urllib.request, urllib.parse, json, os, csv, re, time

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "Tier2_literature")
MANI = os.path.join(ROOT, "manifest.csv")
BIB  = os.path.join(ROOT, "references.bib")
UA   = {"User-Agent": "rag-corpus-builder/1.0 (academic research)"}

def lookup(pmcid):
    """Return (journal, authors, doi) best-effort for a PMCID."""
    q = f"pmcid:{pmcid}"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": q, "format": "json", "pageSize": 1, "resultType": "core"})
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        res = d.get("resultList", {}).get("result", [])
        if not res:
            return None, None, None
        a = res[0]
        ji = a.get("journalInfo") or {}
        j = ji.get("journal") or {}
        journal = j.get("medlineAbbreviation") or j.get("title") or ""
        # authors: prefer authorString, else build from authorList
        auth = a.get("authorString")
        if not auth:
            al = a.get("authorList", {}).get("author", []) if a.get("authorList") else []
            names = [x.get("fullName") for x in al if x.get("fullName")]
            auth = ", ".join(names)
        doi = a.get("doi") or ""
        return journal.strip(), (auth or "").strip().rstrip("."), (doi or "").strip()
    except Exception:
        return None, None, None

# ---- load manifest ----
with open(MANI, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows)} entries from manifest.")

fixed_j = fixed_a = fixed_d = 0
for i, row in enumerate(rows, 1):
    pmcid = row["pmcid"]
    j, a, d = lookup(pmcid)
    time.sleep(0.08)
    if j and not row.get("journal"):
        row["journal"] = j; fixed_j += 1
    if d and not row.get("doi"):
        row["doi"] = d; fixed_d += 1
    # remember best author/journal/doi for md+bib patching
    row["_journal"] = j or row.get("journal") or ""
    row["_authors"] = a or ""
    row["_doi"]     = d or row.get("doi") or ""
    if i % 25 == 0:
        print(f"  looked up {i}/{len(rows)} ...")

print(f"\nFilled: journal={fixed_j}, doi={fixed_d} (authors re-derived where missing)")

# ---- rewrite manifest.csv ----
with open(MANI, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pmcid","pmid","doi","bucket","year","journal","license","text_chars","file"])
    for r in rows:
        w.writerow([r["pmcid"], r.get("pmid",""), r["_doi"], r["bucket"], r["year"],
                    r["_journal"], r.get("license",""), r["text_chars"], r["file"]])
print("manifest.csv rewritten.")

# ---- patch markdown headers + rebuild bib ----
bib_lines = ["% Tier-2 RAG corpus — auto-generated from Europe PMC (Open Access)\n\n"]
patched = 0
for r in rows:
    bucket_dir = os.path.join(ROOT, r["bucket"])
    fpath = os.path.join(bucket_dir, r["file"])
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            md = f.read()
        # extract title from the markdown "# Title" header line
        m = re.search(r"^# (.+)$", md, re.M)
        r["_title"] = m.group(1).strip() if m else ""
        def repl(line_label, value):
            return re.sub(rf"(\*\*{line_label}:\*\*)[^\n]*", rf"\1 {value}", md, count=1)
        md = repl("Journal", r["_journal"] or "(n/a)")
        if r["_authors"]:
            md = repl("Authors", r["_authors"])
        if r["_doi"]:
            md = repl("DOI", r["_doi"])
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(md)
        patched += 1
    # bib entry (rebuilt from enriched fields)
    key = r["pmcid"]
    title = ""
    # pull title from md if needed
    bib_lines.append(
        f"@article{{{key},\n"
        f"  title   = {{{r.get('_title','')}}},\n"
        f"  author  = {{{r['_authors']}}},\n"
        f"  journal = {{{r['_journal']}}},\n"
        f"  year    = {{{r['year']}}},\n"
        f"  doi     = {{{r['_doi']}}},\n"
        f"  url     = {{https://europepmc.org/article/PMC/{key.replace('PMC','')}}}\n}}\n\n")

with open(BIB, "w", encoding="utf-8") as f:
    f.writelines(bib_lines)
print(f"Patched {patched} markdown headers; references.bib rebuilt.")
print("DONE.")
