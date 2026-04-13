import time
import requests
from collections import defaultdict

UNIPROT = "https://rest.uniprot.org"

# ---------- read your project mapping: SYMBOL -> set(UNIPROT) ----------
def load_symbol_to_uniprot(path):
    m = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sym, acc = line.split()[:2]
            m[sym].add(acc)
    return m

# ---------- read GMT (symbol-based) ----------
def read_gmt(path):
    gmt = []
    all_syms = set()
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            name, desc, genes = parts[0], parts[1], parts[2:]
            gmt.append((name, desc, genes))
            all_syms.update(genes)
    return gmt, sorted(all_syms)

# ---------- UniProt mapping (Gene_Name -> UniProtKB) ----------
def submit_mapping(from_db: str, to_db: str, ids):
    r = requests.post(
        f"{UNIPROT}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids), "taxId": "9606"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["jobId"]

def wait_job(job_id: str, sleep_s=2):
    while True:
        r = requests.get(f"{UNIPROT}/idmapping/status/{job_id}", timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("jobStatus") == "RUNNING":
            time.sleep(sleep_s)
            continue
        return j

def fetch_results_json(job_id: str):
    r = requests.get(
        f"{UNIPROT}/idmapping/stream/{job_id}",
        params={"format": "json"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()

def merge_api_results_into_mapping(sym2uni, json_data, tax_id=9606):
    """
    tax id 9606 is the homo sapien taxonomy id
    """
    for item in json_data.get("results", []):
        sym = item.get("from")
        acc = item.get("to", {})
        if sym and acc:
            sym2uni[sym].add(acc)


def pick_one_accession(accessions):
    # Prefer “Swiss-Prot-looking” accessions if present (heuristic).
    # Swiss-Prot accessions are often 6 chars and start with O/P/Q (not always, but common).
    accs = sorted(accessions)
    preferred = [a for a in accs if len(a) == 6 and a[0] in {"O", "P", "Q"}]
    return preferred[0] if preferred else accs[0]


def write_gmt_uniprot(gmt, sym2uni, out_path, mode="one"):
    """
    mode="one"  -> pick one UniProt per symbol
    mode="all"  -> keep all UniProt per symbol
    """
    with open(out_path, "w") as out:
        for name, desc, symbols in gmt:
            uniprots = []
            for s in symbols:
                hits = sym2uni.get(s, set())
                if not hits:
                    continue
                if mode == "all":
                    uniprots.extend(sorted(hits))
                else:
                    uniprots.append(pick_one_accession(hits))
            # remove duplicates
            uniprots = sorted(set(uniprots))
            out.write("\t".join([name, desc] + uniprots) + "\n")


if __name__ == "__main__":
    mapping_file = "data/TCGA_complete_name2uniprotkb.tsv"
    hallmark_gmt = "data/h.all.v2026.1.Hs.symbols.gmt"
    reactome_gmt = "data/reactomes.gmt"

    in_gmt = reactome_gmt
    # out_gmt = "data/hallmark_v2026_1_Hs_uniprot.gmt"
    out_gmt = "data/reactomes_uniprot.gmt"
    sym2uni = load_symbol_to_uniprot(mapping_file)
    gmt, all_syms = read_gmt(in_gmt)

    missing = [s for s in all_syms if s not in sym2uni]
    print("Input Gmt symbols:", len(all_syms))
    print("Missing from project mapping:", len(missing))

    if missing:
        # Hallmark is small enough for one go, no need for batches yet.
        job = submit_mapping(from_db="Gene_Name", to_db="UniProtKB", ids=missing)
        wait_job(job)
        js = fetch_results_json(job)
        merge_api_results_into_mapping(sym2uni, js, tax_id=9606)

    write_gmt_uniprot(gmt, sym2uni, out_gmt, mode="one")
    print("Wrote:", out_gmt)
