import os
from dotenv import load_dotenv
load_dotenv()

"""
Extract symptoms from full-text PubMed/PMC case reports using a local Ollama LLM.

Features:
- Input rare disease string and desired number of reports.
- Searches PubMed for case reports.
- Links PubMed records to PMC full text.
- Optionally filters to PMC Open Access.
- Retrieves full-text PMC XML.
- Sections the article and prioritizes patient/case sections.
- Chunks long text before sending to Ollama.
- Uses Ollama to extract patient-specific symptoms, negative findings, tests,
  suspected diagnoses, and confirmed final diagnosis.
- Normalizes symptoms to HPO terms using NLM Clinical Tables HPO API if possible.
- Outputs structured JSON.

Example:
    python pubmed_pmc_ollama_extractor.py \
        --disease "hereditary angioedema" \
        --n 5 \
        --output hae_cases.json \
        --email your_email@example.com \
        --ollama-model llama3.1 \
        --open-access-only

Important:
- Treat the LLM output as first-pass annotation, not ground truth.
- Manually review evidence sentences before using in a benchmark.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm


# -----------------------------
# API constants
# -----------------------------

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
HPO_API_BASE = "https://clinicaltables.nlm.nih.gov/api/hpo/v3/search"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# -----------------------------
# HTTP utilities
# -----------------------------

def get_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    sleep_sec: float = 0.7,
    timeout: int = 45,
) -> requests.Response:
    last_exc = None

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(sleep_sec * (attempt + 1))

    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_exc


def post_with_retry(
    url: str,
    payload: Dict[str, Any],
    retries: int = 2,
    sleep_sec: float = 0.7,
    timeout: int = 180,
) -> requests.Response:
    last_exc = None

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(sleep_sec * (attempt + 1))

    raise RuntimeError(f"POST failed after {retries} attempts: {url}") from last_exc


# -----------------------------
# PubMed / PMC retrieval
# -----------------------------

def search_pubmed_case_reports(
    disease: str,
    desired_count: int,
    email: str,
    api_key: Optional[str] = None,
    tool: str = "pmc_ollama_case_extractor",
) -> List[str]:
    """
    Search PubMed for case reports related to the disease.

    We request more than desired_count because many PubMed results will not have
    available PMC full text.
    """

    query = (
        f'"{disease}"[Title/Abstract] '
        f'AND ("Case Reports"[Publication Type] OR "case report"[Title/Abstract]) '
        f'AND humans[MeSH Terms]'
    )

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max(desired_count * 8, 40),
        "sort": "relevance",
        "email": email,
        "tool": tool,
    }

    if api_key:
        params["api_key"] = api_key

    response = get_with_retry(f"{NCBI_EUTILS_BASE}/esearch.fcgi", params=params)
    data = response.json()

    result = data.get("esearchresult", {})
    id_list = result.get("idlist", [])
    total_count = int(result.get("count", 0))

    print(f"PubMed total matches for this query: {total_count}")
    print(f"Retrieved first {len(id_list)} candidate IDs for screening.")

    return id_list


def link_pmids_to_pmcids(
    pmids: List[str],
    email: str,
    api_key: Optional[str] = None,
    tool: str = "pmc_ollama_case_extractor",
) -> Dict[str, str]:
    """
    Convert PubMed IDs to PMC IDs using ELink.

    Returns:
        {pmid: pmcid}
    """

    if not pmids:
        return {}

    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": email,
        "tool": tool,
    }

    if api_key:
        params["api_key"] = api_key

    response = get_with_retry(f"{NCBI_EUTILS_BASE}/elink.fcgi", params=params)
    root = ET.fromstring(response.text)

    mapping: Dict[str, str] = {}

    for linkset in root.findall(".//LinkSet"):
        pmid_element = linkset.find("./IdList/Id")
        if pmid_element is None or not pmid_element.text:
            continue

        pmid = pmid_element.text

        link_ids = linkset.findall(".//LinkSetDb/Link/Id")
        if not link_ids:
            continue

        # NCBI returns PMC numeric IDs. The PMCID string is "PMC" + numeric ID.
        pmc_numeric_id = link_ids[0].text
        if pmc_numeric_id:
            mapping[pmid] = f"PMC{pmc_numeric_id}"

    return mapping


def check_pmc_open_access(pmcid: str) -> Optional[Dict[str, Any]]:
    """
    Check if an article is in the PMC Open Access subset.

    Returns OA metadata if present, otherwise None.
    """

    response = get_with_retry(PMC_OA_BASE, params={"id": pmcid})
    root = ET.fromstring(response.text)

    error = root.find(".//error")
    if error is not None:
        return None

    record = root.find(".//record")
    if record is None:
        return None

    links = []
    for link in record.findall(".//link"):
        links.append(
            {
                "format": link.attrib.get("format"),
                "href": link.attrib.get("href"),
                "updated": link.attrib.get("updated"),
            }
        )

    return {
        "pmcid": pmcid,
        "license": record.attrib.get("license"),
        "retracted": record.attrib.get("retracted"),
        "oa_links": links,
    }


def fetch_pmc_xml(
    pmcid: str,
    email: str,
    api_key: Optional[str] = None,
    tool: str = "pmc_ollama_case_extractor",
) -> str:
    """
    Fetch PMC full-text XML using EFetch.
    """

    pmc_numeric = pmcid.replace("PMC", "")

    params = {
        "db": "pmc",
        "id": pmc_numeric,
        "retmode": "xml",
        "email": email,
        "tool": tool,
    }

    if api_key:
        params["api_key"] = api_key

    response = get_with_retry(f"{NCBI_EUTILS_BASE}/efetch.fcgi", params=params)
    return response.text


# -----------------------------
# XML parsing and sectioning
# -----------------------------

def element_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""

    return normalize_whitespace(" ".join(t.strip() for t in element.itertext() if t.strip()))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_pmc_article_xml(xml_text: str) -> Dict[str, Any]:
    """
    Extract only year, abstract, sections, and body text from PMC XML.

    We keep sections/body internally so the LLM can still extract symptoms,
    but we only preserve year and abstract in the final JSON metadata.
    """

    root = ET.fromstring(xml_text)

    year = ""
    year_el = root.find(".//pub-date/year")
    if year_el is not None and year_el.text:
        year = year_el.text.strip()

    abstract = element_text(root.find(".//abstract"))

    sections = []
    for sec in root.findall(".//body//sec"):
        title = element_text(sec.find("title"))
        text = element_text(sec)

        if title and text.lower().startswith(title.lower()):
            text = text[len(title):].strip()

        if text:
            sections.append(
                {
                    "section_title": title,
                    "section_text": text,
                    "priority": section_priority(title),
                }
            )

    sections = sorted(sections, key=lambda s: s["priority"], reverse=True)

    body_text = element_text(root.find(".//body"))

    return {
        "year": year,
        "abstract": abstract,
        "sections": sections,
        "body_text": body_text,
    }


def section_priority(title: str) -> int:
    """
    Prioritize sections likely to contain patient-level facts.
    """

    t = title.lower()

    high = [
        "case presentation",
        "case report",
        "case history",
        "clinical presentation",
        "patient presentation",
        "clinical findings",
        "history",
    ]

    medium = [
        "case",
        "patient",
        "investigation",
        "investigations",
        "clinical course",
        "diagnosis",
        "treatment",
        "methods",
    ]

    low = [
        "discussion",
        "background",
        "introduction",
        "conclusion",
        "references",
    ]

    if any(k in t for k in high):
        return 100
    if any(k in t for k in medium):
        return 60
    if any(k in t for k in low):
        return 5

    return 20


def choose_sections_for_llm(article: Dict[str, Any], max_sections: int = 8) -> List[Dict[str, str]]:
    """
    Select the most patient-relevant sections.

    If no useful sections are found, falls back to abstract and body preview.
    """

    selected = []

    for sec in article.get("sections", []):
        if sec["priority"] >= 20 and len(sec["section_text"]) > 100:
            selected.append(
                {
                    "section_title": sec["section_title"] or "Untitled section",
                    "section_text": sec["section_text"],
                }
            )

        if len(selected) >= max_sections:
            break

    if selected:
        return selected

    fallback = f"ABSTRACT: {article.get('abstract', '')}\n\nBODY: {article.get('body_text', '')}"
    return [{"section_title": "Fallback abstract/body", "section_text": fallback}]


def chunk_text(text: str, max_chars: int = 3000, overlap_chars: int = 500) -> List[str]:
    """
    Chunk text by character count with overlap.

    Ollama context windows vary by model, so this keeps chunks conservative.
    """

    text = normalize_whitespace(text)

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]

        # Try to cut at a sentence boundary near the end.
        boundary = max(chunk.rfind(". "), chunk.rfind("; "), chunk.rfind("\n"))
        if boundary > int(max_chars * 0.65):
            chunk = chunk[:boundary + 1]
            end = start + boundary + 1

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - overlap_chars)

    return chunks


# -----------------------------
# Ollama extraction
# -----------------------------

def build_extraction_prompt(
    disease: str,
    pmcid: str,
    section_title: str,
    chunk_index: int,
    chunk_count: int,
    text_chunk: str,
) -> str:
    """
    Strict extraction prompt for patient-specific facts.
    """

    return f"""
You are extracting patient-specific clinical information from a full-text medical case report.

Disease query: {disease}
PMCID: {pmcid}
Section title: {section_title}
Chunk: {chunk_index + 1} of {chunk_count}

TASK:
Extract only patient-specific clinical facts explicitly stated in the text.

STRICT RULES:
1. Do NOT infer symptoms from the disease name.
2. Do NOT extract general background information about the disease.
3. Do NOT extract information from sentences that describe disease features generally unless tied to the actual patient.
4. Every extracted item must include a short evidence sentence copied from the text.
5. Separate symptoms/signs from diagnostic tests.
6. Handle negation carefully. For example, "no fever" should go under negative_findings, not symptoms_presented.
7. If multiple patients are described, create separate patient records.
8. Return valid JSON only. No markdown. No commentary.

OUTPUT SCHEMA:
{{
  "patients": [
    {{
      "patient_id": "case_1",
      "age": "",
      "sex": "",
      "confirmed_diagnosis": "",
      "final_diagnosis_evidence": "",
      "symptoms_presented": [
        {{
          "raw_symptom": "",
          "status": "present",
          "temporality": "",
          "evidence_sentence": ""
        }}
      ],
      "negative_findings": [
        {{
          "finding": "",
          "status": "absent",
          "evidence_sentence": ""
        }}
      ],
      "diagnostic_tests": [
        {{
          "test": "",
          "result": "",
          "evidence_sentence": ""
        }}
      ],
      "initial_misdiagnoses_or_mimics": [
        {{
          "diagnosis": "",
          "evidence_sentence": ""
        }}
      ]
    }}
  ]
}}

TEXT:
{text_chunk}
""".strip()


def call_ollama_extract(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    ollama_url: str = OLLAMA_CHAT_URL,
) -> Dict[str, Any]:
    """
    Call local Ollama and return parsed JSON.

    Uses format=json to encourage valid JSON output.
    """

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful biomedical information extraction assistant. "
                    "You return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }

    response = post_with_retry(ollama_url, payload=payload)
    data = response.json()

    content = data.get("message", {}).get("content", "")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object from response.
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"Ollama did not return valid JSON:\n{content[:1000]}")

'''
# -----------------------------
# HPO normalization
# -----------------------------

def normalize_to_hpo(term: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Normalize a symptom phrase to candidate HPO terms using NLM Clinical Tables HPO API.

    Returns candidate matches:
        [{"hpo_id": "...", "hpo_name": "..."}]
    """

    term = term.strip()
    if not term:
        return []

    params = {
        "terms": term,
        "maxList": max_results,
    }

    try:
        response = get_with_retry(HPO_API_BASE, params=params, retries=2, timeout=20)
        data = response.json()
    except Exception:
        return []

    # NLM Clinical Tables response format is usually:
    # [total_count, codes, extra, display_rows]
    # display_rows often contains rows like [HPO_ID, HPO_NAME].
    candidates = []

    try:
        display_rows = data[3]
        for row in display_rows:
            if len(row) >= 2:
                candidates.append(
                    {
                        "hpo_id": str(row[0]),
                        "hpo_name": str(row[1]),
                    }
                )
    except Exception:
        return []

    return candidates


def attach_hpo_to_extractions(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add HPO candidate terms to each symptom.
    """

    for patient in extraction.get("patients", []):
        symptoms = patient.get("symptoms_presented", [])

        for symptom in symptoms:
            raw = symptom.get("normalized_symptom") or symptom.get("raw_symptom") or ""
            candidates = normalize_to_hpo(raw)

            symptom["hpo_candidates"] = candidates
            if candidates:
                symptom["best_hpo_id"] = candidates[0]["hpo_id"]
                symptom["best_hpo_name"] = candidates[0]["hpo_name"]
            else:
                symptom["best_hpo_id"] = ""
                symptom["best_hpo_name"] = ""

    return extraction
'''

# -----------------------------
# Merge chunk-level extractions
# -----------------------------

def symptom_key(symptom: Dict[str, Any]) -> Tuple[str, str]:
    name = (
        symptom.get("normalized_symptom")
        or symptom.get("raw_symptom")
        or ""
    ).lower().strip()

    evidence = symptom.get("evidence_sentence", "").lower().strip()
    return name, evidence[:120]


def merge_patient_records(extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge patient records from multiple chunks.

    This is intentionally conservative.
    For multi-patient reports, manual review is still recommended.
    """

    merged: Dict[str, Dict[str, Any]] = {}

    for extraction in extractions:
        for patient in extraction.get("patients", []):
            pid = patient.get("patient_id") or "case_1"

            if pid not in merged:
                merged[pid] = {
                    "patient_id": pid,
                    "age": patient.get("age", ""),
                    "sex": patient.get("sex", ""),
                    "confirmed_diagnosis": patient.get("confirmed_diagnosis", ""),
                    "final_diagnosis_evidence": patient.get("final_diagnosis_evidence", ""),
                    "symptoms_presented": [],
                    "negative_findings": [],
                    "diagnostic_tests": [],
                    "initial_misdiagnoses_or_mimics": [],
                }

            target = merged[pid]

            for field in ["age", "sex", "confirmed_diagnosis", "final_diagnosis_evidence"]:
                if not target.get(field) and patient.get(field):
                    target[field] = patient[field]

            append_unique_dicts(
                target["symptoms_presented"],
                patient.get("symptoms_presented", []),
                key_func=symptom_key,
            )

            append_unique_dicts(
                target["negative_findings"],
                patient.get("negative_findings", []),
                key_func=lambda x: (
                    x.get("finding", "").lower().strip(),
                    x.get("evidence_sentence", "").lower().strip()[:120],
                ),
            )

            append_unique_dicts(
                target["diagnostic_tests"],
                patient.get("diagnostic_tests", []),
                key_func=lambda x: (
                    x.get("test", "").lower().strip(),
                    x.get("result", "").lower().strip(),
                    x.get("evidence_sentence", "").lower().strip()[:120],
                ),
            )

            append_unique_dicts(
                target["initial_misdiagnoses_or_mimics"],
                patient.get("initial_misdiagnoses_or_mimics", []),
                key_func=lambda x: (
                    x.get("diagnosis", "").lower().strip(),
                    x.get("evidence_sentence", "").lower().strip()[:120],
                ),
            )

    return list(merged.values())


def append_unique_dicts(
    target: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
    key_func,
) -> None:
    seen = {key_func(item) for item in target}

    for item in incoming:
        key = key_func(item)
        if key not in seen:
            target.append(item)
            seen.add(key)


# -----------------------------
# Main pipeline
# -----------------------------

def process_one_article(
    disease: str,
    pmid: str,
    pmcid: str,
    email: str,
    ollama_model: str,
    api_key: Optional[str] = None,
    open_access_only: bool = False,
    max_chars_per_chunk: int = 7500,
) -> Optional[Dict[str, Any]]:
    """
    Fetch, section, chunk, extract, normalize, and package one PMC article.
    """

    oa_metadata = None

    if open_access_only:
        oa_metadata = check_pmc_open_access(pmcid)
        if oa_metadata is None:
            return None
        if oa_metadata.get("retracted") == "yes":
            return None
    else:
        # Still try to get OA metadata when available.
        try:
            oa_metadata = check_pmc_open_access(pmcid)
        except Exception:
            oa_metadata = None

    xml_text = fetch_pmc_xml(pmcid=pmcid, email=email, api_key=api_key)
    article = parse_pmc_article_xml(xml_text)

    selected_sections = choose_sections_for_llm(article)

    chunk_extractions = []
    llm_input_log = []

    for section in selected_sections:
        chunks = chunk_text(section["section_text"], max_chars=max_chars_per_chunk)

        for i, chunk in enumerate(chunks):
            prompt = build_extraction_prompt(
                disease=disease,
                pmcid=pmcid,
                section_title=section["section_title"],
                chunk_index=i,
                chunk_count=len(chunks),
                text_chunk=chunk,
            )

            try:
                extraction = call_ollama_extract(prompt=prompt, model=ollama_model)
                chunk_extractions.append(extraction)
            except Exception as exc:
                print(f"    Ollama extraction failed for {pmcid}, section {section['section_title']}: {exc}")
                continue

            llm_input_log.append(
                {
                    "section_title": section["section_title"],
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "chunk_characters": len(chunk),
                }
            )

    merged_patients = merge_patient_records(chunk_extractions)
    merged_extraction = {"patients": merged_patients}
    # merged_extraction = attach_hpo_to_extractions(merged_extraction)

    return {
        "pmcid": pmcid,
        "pubmed_url": PUBMED_URL_TEMPLATE.format(pmid=pmid),
        "year": article.get("year", ""),
        "abstract": article.get("abstract", ""),
        "extraction": merged_extraction,
        "review_status": "needs_human_review",
    }


def collect_cases(
    disease: str,
    desired_count: int,
    output_file: str,
    email: str,
    ollama_model: str,
    api_key: Optional[str] = None,
    open_access_only: bool = False,
    max_chars_per_chunk: int = 7500,
) -> Dict[str, Any]:
    """
    Full collection pipeline.
    """

    print(f"Searching PubMed for case reports about: {disease}")
    pmids = search_pubmed_case_reports(
        disease=disease,
        desired_count=desired_count,
        email=email,
        api_key=api_key,
    )
    pmids = pmids + [disease]

    pmid_to_pmcid = link_pmids_to_pmcids(pmids, email=email, api_key=api_key)
    print(f"Found {len(pmid_to_pmcid)} candidate records with PMC full-text links.")

    records = []

    for pmid, pmcid in tqdm(pmid_to_pmcid.items(), desc="Processing PMC articles"):
        if len(records) >= desired_count:
            break

        try:
            record = process_one_article(
                disease=disease,
                pmid=pmid,
                pmcid=pmcid,
                email=email,
                ollama_model=ollama_model,
                api_key=api_key,
                open_access_only=open_access_only,
                max_chars_per_chunk=max_chars_per_chunk,
            )
        except Exception as exc:
            print(f"  Skipping {pmcid}: {exc}")
            continue

        if record is None:
            continue

        records.append(record)

        # Be polite to external APIs.
        time.sleep(0.5)

    output = {
        "query": {
            "disease": disease,
            "desired_report_count": desired_count,
            "retrieved_report_count": len(records),
            "open_access_only": open_access_only,
            "ollama_model": ollama_model,
        },
        "records": records,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} records to {output_file}")
    return output

def extract_case_by_pmid(
    pmid: str,
    disease: str,
    email: str,
    ollama_model: str,
    api_key: Optional[str] = None,
    open_access_only: bool = False,
    max_chars_per_chunk: int = 7500,
) -> Optional[Dict[str, Any]]:
    """
    Extract a single case report using its PubMed ID (PMID).
    Returns the structured extraction or None if not available.
    """
    # Link PMID to PMCID
    pmid_to_pmcid = link_pmids_to_pmcids([pmid], email=email, api_key=api_key)
    pmcid = pmid_to_pmcid.get(pmid)
    if not pmcid:
        print(f"No PMC full-text found for PMID {pmid}")
        return None

    # Process the article as usual
    try:
        record = process_one_article(
            disease=disease,
            pmid=pmid,
            pmcid=pmcid,
            email=email,
            ollama_model=ollama_model,
            api_key=api_key,
            open_access_only=open_access_only,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        return record
    except Exception as exc:
        print(f"Failed to extract case for PMID {pmid}: {exc}")
        return None

# -----------------------------
# CLI
# -----------------------------

def prompt_required(prompt_text: str) -> str:
    """
    Prompt until the user enters a non-empty value.
    """
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        print("This field is required.")


def prompt_int(prompt_text: str, default: Optional[int] = None) -> int:
    """
    Prompt for an integer, optionally with a default.
    """
    while True:
        raw = input(prompt_text).strip()

        if not raw and default is not None:
            return default

        try:
            value = int(raw)
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")


def prompt_yes_no(prompt_text: str, default: bool = False) -> bool:
    """
    Prompt for yes/no input.
    """
    default_text = "Y/n" if default else "y/N"

    while True:
        raw = input(f"{prompt_text} ({default_text}): ").strip().lower()

        if not raw:
            return default

        if raw in {"y", "yes"}:
            return True

        if raw in {"n", "no"}:
            return False

        print("Please enter yes or no.")


def main() -> None:
    print("\nPMC / PubMed Case Report Symptom Extractor")
    print("Uses local Ollama to extract symptoms from available PMC full-text case reports.\n")

    disease = prompt_required("Rare disease name: ")
    desired_count = prompt_int("Number of reports desired: ")
    output_file = "data/" + prompt_required("Output JSON filename, e.g. hae.json: ") + ".json"
    # email = prompt_required("Email for NCBI E-utilities: ")
    email = os.getenv("EMAIL")
    # api_key = input("NCBI API key, optional. Press Enter to skip: ").strip()
    # if not api_key:
    api_key = None

    # ollama_model = input("Ollama model name [llama3.1]: ").strip()
    # if not ollama_model:
    ollama_model = "pubmed-extractor"
    # ollama_model = "pubmed-extractor-7b" 

    open_access_only = prompt_yes_no(
        "Only use PMC Open Access articles?",
        default=True,
    )

    max_chars_per_chunk = prompt_int(
        "Max characters per LLM chunk [3000 recommended]: ",
        default=3000,
    )

    print("\nStarting extraction with the following settings:")
    print(f"  Disease: {disease}")
    print(f"  Desired reports: {desired_count}")
    print(f"  Output file: {output_file}")
    print(f"  Ollama model: {ollama_model}")
    print(f"  Open Access only: {open_access_only}")
    print(f"  Max chars per chunk: {max_chars_per_chunk}")
    print()

    collect_cases(
        disease=disease,
        desired_count=desired_count,
        output_file=output_file,
        email=email,
        ollama_model=ollama_model,
        api_key=api_key,
        open_access_only=open_access_only,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    # extract_case_by_pmid(
    #     pmid="36155286",
    #     disease=disease,
    #     email=email,
    #     ollama_model=ollama_model,
    #     api_key=api_key,
    #     open_access_only=open_access_only,
    #     max_chars_per_chunk=max_chars_per_chunk,
    # )

if __name__ == "__main__":
    main()