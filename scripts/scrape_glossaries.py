#!/usr/bin/env python3
"""
B2B Sales Vocabulary Scraper

Fetches and extracts sales terminology from authoritative glossaries.
Output: data/vocabulary.json
"""

import json
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List
import time

# Glossary sources to scrape
GLOSSARY_URLS = {
    "pclub": "https://www.pclub.io/blog/sales-definitions-glossary",
    "cognism": "https://www.cognism.com/blog/sales-terms",
    "kalungi": "https://www.kalungi.com/blog/saas-sales-marketing-acronyms-abbreviations",
    "walnut": "https://www.walnut.io/blog/sales-tips/sales-terminology/",
    "leadiq": "https://leadiq.com/blog/b2b-saas-sales-glossary-and-acronym-cheat-sheet",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_page(url: str) -> str:
    """Fetch HTML content from URL"""
    print(f"  Fetching: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return ""


def extract_terms_generic(html: str) -> List[Dict]:
    """Generic term extraction from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    terms = []

    # Look for definition lists, bold terms, or heading + paragraph patterns
    # Pattern 1: <strong>Term</strong> or <b>Term</b> followed by text
    for bold in soup.find_all(["strong", "b"]):
        term = bold.get_text().strip()
        # Skip if too long (probably not a term)
        if len(term) > 50 or len(term) < 2:
            continue
        # Get surrounding text as definition
        parent = bold.parent
        if parent:
            full_text = parent.get_text()
            # Extract definition after the term
            if term in full_text:
                definition = full_text.split(term, 1)[-1].strip()
                definition = re.sub(r'^[:\-–—]\s*', '', definition)  # Remove leading punctuation
                if len(definition) > 10:
                    terms.append({
                        "term": term,
                        "definition": definition[:500]  # Truncate long definitions
                    })

    # Pattern 2: <h2>/<h3>/<h4> followed by <p>
    for heading in soup.find_all(["h2", "h3", "h4"]):
        term = heading.get_text().strip()
        if len(term) > 50 or len(term) < 2:
            continue
        # Get next sibling paragraph
        next_elem = heading.find_next_sibling()
        if next_elem and next_elem.name == "p":
            definition = next_elem.get_text().strip()
            if len(definition) > 10:
                terms.append({
                    "term": term,
                    "definition": definition[:500]
                })

    # Deduplicate by term name
    seen = set()
    unique_terms = []
    for t in terms:
        term_lower = t["term"].lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(t)

    return unique_terms


def categorize_term(term: str, definition: str) -> str:
    """Categorize a term based on keywords"""
    text = (term + " " + definition).lower()

    if any(w in text for w in ["quota", "attainment", "target", "number", "goal"]):
        return "quota_forecast"
    elif any(w in text for w in ["pipeline", "funnel", "stage", "deal", "opportunity"]):
        return "pipeline_deals"
    elif any(w in text for w in ["mql", "sql", "lead", "prospect", "qualification"]):
        return "lead_qualification"
    elif any(w in text for w in ["outbound", "inbound", "expansion", "renewal", "churn"]):
        return "gtm_motion"
    elif any(w in text for w in ["arr", "mrr", "cac", "ltv", "nrr", "revenue", "metric"]):
        return "metrics_kpi"
    elif any(w in text for w in ["sdr", "bdr", "ae", "csm", "rep", "manager"]):
        return "roles"
    elif any(w in text for w in ["meddic", "bant", "spin", "challenger", "methodology"]):
        return "methodology"
    else:
        return "general"


def scrape_all_glossaries() -> Dict:
    """Scrape all glossary sources and combine results"""
    all_terms = []

    for source_name, url in GLOSSARY_URLS.items():
        print(f"\nScraping {source_name}...")
        html = fetch_page(url)
        if html:
            terms = extract_terms_generic(html)
            for t in terms:
                t["source"] = source_name
                t["category"] = categorize_term(t["term"], t["definition"])
            all_terms.extend(terms)
            print(f"  Found {len(terms)} terms")
        time.sleep(1)  # Be nice to servers

    # Deduplicate across sources
    seen = set()
    unique_terms = []
    for t in all_terms:
        term_lower = t["term"].lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(t)

    # Group by category
    by_category = {}
    for t in unique_terms:
        cat = t["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(t)

    return {
        "total_terms": len(unique_terms),
        "by_category": by_category,
        "all_terms": unique_terms
    }


def add_manual_vocabulary() -> List[Dict]:
    """Add manually curated B2B vocabulary that may not be in glossaries"""
    manual_terms = [
        # Quota/Forecast colloquialisms
        {"term": "my number", "definition": "A sales rep's quarterly or annual quota target", "category": "quota_forecast"},
        {"term": "hit my number", "definition": "Achieve one's sales quota target", "category": "quota_forecast"},
        {"term": "make quota", "definition": "Achieve assigned sales target", "category": "quota_forecast"},
        {"term": "miss quota", "definition": "Fail to achieve sales target", "category": "quota_forecast"},
        {"term": "above/below number", "definition": "Performance relative to quota", "category": "quota_forecast"},
        {"term": "gap to quota", "definition": "Difference between current forecast and quota target", "category": "quota_forecast"},
        {"term": "percent to goal", "definition": "Current attainment as percentage of quota", "category": "quota_forecast"},

        # Pipeline colloquialisms
        {"term": "pipeline review", "definition": "Recurring meeting between sales rep and manager to review deal status and forecast", "category": "pipeline_deals"},
        {"term": "pipe", "definition": "Short for pipeline - active sales opportunities", "category": "pipeline_deals"},
        {"term": "build pipe", "definition": "Generate new pipeline/opportunities", "category": "pipeline_deals"},
        {"term": "my book", "definition": "A rep's assigned accounts or territory", "category": "pipeline_deals"},
        {"term": "book of business", "definition": "Portfolio of accounts assigned to a rep", "category": "pipeline_deals"},
        {"term": "deal slip", "definition": "When a deal's expected close date moves to a later period", "category": "pipeline_deals"},
        {"term": "push deal", "definition": "Move deal close date to next period", "category": "pipeline_deals"},
        {"term": "pull deal", "definition": "Accelerate deal to close earlier", "category": "pipeline_deals"},
        {"term": "stuck deal", "definition": "Deal not progressing through stages", "category": "pipeline_deals"},
        {"term": "stalled opportunity", "definition": "Deal with no recent activity or progression", "category": "pipeline_deals"},

        # Account status
        {"term": "green account", "definition": "Healthy customer account with positive engagement", "category": "account_health"},
        {"term": "red account", "definition": "At-risk customer showing churn signals", "category": "account_health"},
        {"term": "yellow account", "definition": "Customer account needing attention but not critical", "category": "account_health"},
        {"term": "logo", "definition": "A customer, counted as unit (e.g., 'new logos' = new customers)", "category": "account_health"},
        {"term": "new logo", "definition": "Net-new customer acquisition", "category": "account_health"},

        # GTM motion phrases
        {"term": "land and expand", "definition": "Strategy of starting small and growing within account", "category": "gtm_motion"},
        {"term": "whitespace", "definition": "Untapped opportunity within existing customer", "category": "gtm_motion"},
        {"term": "greenfield", "definition": "Net-new prospect with no existing relationship", "category": "gtm_motion"},
        {"term": "net-new", "definition": "Brand new customer or opportunity", "category": "gtm_motion"},
        {"term": "run rate", "definition": "Current revenue extrapolated to annual rate", "category": "gtm_motion"},

        # Compensation/Commission
        {"term": "OTE", "definition": "On-Target Earnings - expected total compensation if quota achieved", "category": "compensation"},
        {"term": "accelerators", "definition": "Higher commission rate for performance above quota", "category": "compensation"},
        {"term": "spiff", "definition": "Short-term bonus for specific sales behaviors", "category": "compensation"},
        {"term": "kicker", "definition": "Additional bonus for exceptional performance", "category": "compensation"},

        # Meeting/Activity terms
        {"term": "QBR", "definition": "Quarterly Business Review - strategic customer meeting", "category": "meetings"},
        {"term": "EBR", "definition": "Executive Business Review - C-level customer meeting", "category": "meetings"},
        {"term": "disco call", "definition": "Discovery call to understand prospect needs", "category": "meetings"},
        {"term": "demo", "definition": "Product demonstration to prospect", "category": "meetings"},
        {"term": "POC", "definition": "Proof of Concept - trial implementation", "category": "meetings"},
        {"term": "POV", "definition": "Proof of Value - demonstrating ROI", "category": "meetings"},

        # Forecast categories
        {"term": "commit", "definition": "Deals sales rep is confident will close this period", "category": "forecast"},
        {"term": "best case", "definition": "Optimistic forecast including probable deals", "category": "forecast"},
        {"term": "upside", "definition": "Potential deals beyond best case", "category": "forecast"},
        {"term": "worst case", "definition": "Conservative forecast of minimum expected", "category": "forecast"},
        {"term": "pipeline coverage", "definition": "Ratio of pipeline to quota (e.g., 3x coverage)", "category": "forecast"},

        # Activity metrics
        {"term": "touches", "definition": "Number of outreach attempts to prospect", "category": "activity"},
        {"term": "sequence", "definition": "Automated series of outreach emails", "category": "activity"},
        {"term": "cadence", "definition": "Rhythm and timing of sales outreach", "category": "activity"},
        {"term": "connect rate", "definition": "Percentage of calls that reach a person", "category": "activity"},
        {"term": "no-show", "definition": "Prospect who misses scheduled meeting", "category": "activity"},
    ]

    for t in manual_terms:
        t["source"] = "manual"

    return manual_terms


def main():
    print("=" * 60)
    print("B2B VOCABULARY SCRAPER")
    print("=" * 60)

    # Scrape online glossaries
    print("\nPhase 1: Scraping online glossaries...")
    scraped = scrape_all_glossaries()

    # Add manual vocabulary
    print("\nPhase 2: Adding manual vocabulary...")
    manual = add_manual_vocabulary()
    print(f"  Added {len(manual)} manual terms")

    # Combine
    all_terms = scraped["all_terms"] + manual

    # Recategorize and count
    by_category = {}
    for t in all_terms:
        cat = t["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(t)

    result = {
        "total_terms": len(all_terms),
        "categories": {k: len(v) for k, v in by_category.items()},
        "by_category": by_category,
        "all_terms": all_terms
    }

    # Save to file
    output_path = "data/vocabulary.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"RESULTS")
    print("=" * 60)
    print(f"Total terms: {result['total_terms']}")
    print(f"\nBy category:")
    for cat, count in sorted(result["categories"].items()):
        print(f"  {cat}: {count}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
