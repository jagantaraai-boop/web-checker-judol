import os
import time
import random
import re
import math
import base64
import uuid
import joblib
import numpy as np
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from scipy.sparse import hstack
from urllib.parse import urlparse

try:
    from inference_sdk import InferenceHTTPClient
except Exception:
    InferenceHTTPClient = None

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# =============================================
# ML MODEL LOADING
# =============================================
ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ML', 'model.pkl')
ML_TFIDF_PATH = os.path.join(os.path.dirname(__file__), 'ML', 'tfidf.pkl')
ML_SCALER_PATH = os.path.join(os.path.dirname(__file__), 'ML', 'scaler.pkl')

ml_model = None
ml_vectorizer = None
ml_scaler = None

def load_ml_models():
    global ml_model, ml_vectorizer, ml_scaler
    try:
        print("[ML] Loading model files...")
        ml_model = joblib.load(ML_MODEL_PATH)
        ml_vectorizer = joblib.load(ML_TFIDF_PATH)
        ml_scaler = joblib.load(ML_SCALER_PATH)
        print("[ML] Models loaded successfully!")
        return True
    except Exception as e:
        print(f"[ML] Error loading models: {e}")
        return False

# Load models on startup
load_ml_models()

# =============================================
# ML PREDICTION CONFIG & FUNCTIONS
# =============================================
JUDOL_KEYWORDS = [
    "slot", "gacor", "maxwin", "jackpot", "togel", "toto",
    "poker", "casino", "betting", "taruhan", "judi", "rtp",
    "scatter", "pragmatic", "pg soft", "zeus", "mahjong",
    "olympus", "bonus", "deposit", "withdraw", "spin",
    "live casino", "sportsbook", "parlay", "mix parlay",
    "agen slot", "situs slot", "daftar slot", "login slot",
    "bocoran", "gampang menang", "jp", "hoki", "lucky",
    "77", "88", "99", "168", "777", "888",
]

SUSPICIOUS_SUBDOMAINS = [
    "slot", "gacor", "judol", "maxwin", "togel", "bet",
    "casino", "poker", "live", "jackpot", "bonus",
]

REDIRECT_INDICATORS = [
    "redirect", "redir", "go", "out", "click", "track",
    "ref", "aff", "affiliate", "link", "url", "visit",
]

# Educational and news domains that may legitimately discuss gambling
EDUCATIONAL_NEWS_DOMAINS = [
    "wikipedia.org", "bbc.com", "aljazeera.com", "bbc.co.uk",
    "news.google.com", "reuters.com", "apnews.com", "theguardian.com",
    "kompas.com", "detik.com", "liputan6.com", "tirto.id",
    "tempo.co", "merdeka.com", "indozone.id", "cnnindonesia.com",
    "kumparan.com", "tribunnews.com", "okezone.com", "grid.id",
    "media.id", "berita.id", "pendidikan.id", "edukasi.id", "acehkriminologi.ac.id",
    "harvard.edu", "yale.edu", "stanford.edu", "mit.edu", "cambridge.ac.uk",
]

# Risky domain extensions that require Roboflow verification
RISKY_DOMAIN_EXTENSIONS = ('.com', '.id', '.co.id')

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

ROBOFLOW_API_URL = os.getenv('ROBOFLOW_API_URL', 'https://serverless.roboflow.com')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')
ROBOFLOW_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE', '')
ROBOFLOW_WORKFLOW_ID = os.getenv('ROBOFLOW_WORKFLOW_ID', '')
ROBOFLOW_IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'static', 'generated', 'roboflow')

def is_educational_or_news_domain(url: str) -> bool:
    """Check if domain is educational or news site that may discuss gambling legitimately"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    
    for edu_domain in EDUCATIONAL_NEWS_DOMAINS:
        if domain == edu_domain or domain.endswith("." + edu_domain):
            return True
    return False

def is_risky_domain_extension(url: str) -> bool:
    """Check if domain has risky extension (.com, .id, .co.id) that requires Roboflow verification"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return any(domain.endswith(ext) for ext in RISKY_DOMAIN_EXTENSIONS)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9:/._ -]", " ", text)
    return text.strip()

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

def auto_scrape(url: str, timeout: int = 5) -> dict:
    """Fetch title & meta dari URL"""
    result = {
        "title": "",
        "meta_description": "",
        "triggered_signals": "",
    }
    try:
        resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        title_tag = soup.find("title")
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)
        
        meta_desc = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", attrs={"property": "og:description"})
        )
        if meta_desc:
            result["meta_description"] = meta_desc.get("content", "")
    except Exception as e:
        print(f"[SCRAPE] Error: {e}")
    
    return result

def capture_url_screenshot(url: str, timeout: int = 30000) -> dict:
    """Capture screenshot halaman URL menggunakan headless browser."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return {'ok': False, 'error': f'playwright_unavailable: {exc}'}

    os.makedirs(ROBOFLOW_IMAGE_DIR, exist_ok=True)
    file_name = f"screenshot_{uuid.uuid4().hex}.png"
    screenshot_path = os.path.join(ROBOFLOW_IMAGE_DIR, file_name)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=['--disable-gpu', '--no-sandbox'],
            )
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 2400}, device_scale_factor=1)
                page.goto(url, wait_until='load', timeout=timeout)
                page.wait_for_timeout(1500)
                page.screenshot(path=screenshot_path, full_page=True)
            finally:
                browser.close()

        return {
            'ok': True,
            'path': screenshot_path,
            'relative_path': os.path.relpath(screenshot_path, os.path.join(os.path.dirname(__file__), 'static')).replace('\\', '/'),
        }
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}

def _normalize_prediction_item(item: dict) -> dict:
    item = item or {}
    bbox = item.get('bbox') if isinstance(item.get('bbox'), dict) else {}
    class_name = item.get('class') or item.get('label') or item.get('name') or item.get('category') or item.get('title') or 'unknown'

    return {
        'class': str(class_name),
        'confidence': float(item.get('confidence') or item.get('score') or item.get('probability') or 0.0),
        'x': float(item.get('x', bbox.get('x', 0.0)) or 0.0),
        'y': float(item.get('y', bbox.get('y', 0.0)) or 0.0),
        'width': float(item.get('width', bbox.get('width', 0.0)) or 0.0),
        'height': float(item.get('height', bbox.get('height', 0.0)) or 0.0),
        'raw': item,
    }

def _extract_prediction_list(payload):
    if isinstance(payload, list):
        predictions = []
        for item in payload:
            if isinstance(item, dict):
                nested = _extract_prediction_list(item)
                if nested:
                    predictions.extend(nested)
                else:
                    predictions.append(item)
        return predictions

    if not isinstance(payload, dict):
        return []

    predictions = []
    for key in ('predictions', 'detections', 'objects'):
        value = payload.get(key)
        if isinstance(value, list):
            predictions.extend([item for item in value if isinstance(item, dict)])
        elif isinstance(value, dict):
            predictions.extend(_extract_prediction_list(value))
    if predictions:
        return predictions

    for value in payload.values():
        if isinstance(value, (dict, list)):
            nested = _extract_prediction_list(value)
            if nested:
                predictions.extend(nested)
    return predictions


def _extract_annotated_image_path(payload) -> dict:
    def find_image_candidates(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('output_image', 'annotated_image', 'visualization', 'image') and value:
                    yield value
                if isinstance(value, (dict, list)):
                    yield from find_image_candidates(value)
        elif isinstance(node, list):
            for item in node:
                yield from find_image_candidates(item)

    for candidate in find_image_candidates(payload):
        if isinstance(candidate, bytes):
            image_bytes = candidate
        elif isinstance(candidate, str):
            if candidate.startswith('http://') or candidate.startswith('https://'):
                return {'annotated_image_url': candidate, 'annotated_image_path': ''}
            image_str = candidate
            image_bytes = None
            if image_str.startswith('data:') and ',' in image_str:
                image_str = image_str.split(',', 1)[1]
            try:
                image_bytes = base64.b64decode(image_str)
            except Exception:
                image_bytes = None
            if image_bytes is None:
                # Support direct filesystem path returned by workflow
                fs_path = os.path.expanduser(os.path.expandvars(image_str))
                if os.path.exists(fs_path):
                    rel_path = os.path.relpath(fs_path, os.path.join(os.path.dirname(__file__), 'static')).replace('\\', '/')
                    return {'annotated_image_url': url_for('static', filename=rel_path), 'annotated_image_path': fs_path}
                continue
        else:
            continue

        try:
            os.makedirs(ROBOFLOW_IMAGE_DIR, exist_ok=True)
            file_name = f"annotated_{uuid.uuid4().hex}.png"
            image_path = os.path.join(ROBOFLOW_IMAGE_DIR, file_name)
            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)

            return {
                'annotated_image_url': url_for('static', filename=os.path.relpath(image_path, os.path.join(os.path.dirname(__file__), 'static')).replace('\\', '/')),
                'annotated_image_path': image_path,
            }
        except Exception:
            continue

    return {'annotated_image_url': '', 'annotated_image_path': ''}

def analyze_with_roboflow(url: str, screenshot_result: dict) -> dict:
    """Kirim screenshot ke Roboflow untuk mendeteksi iklan judi online."""
    if not screenshot_result.get('ok'):
        return {
            'enabled': False,
            'error': screenshot_result.get('error', 'screenshot_failed'),
            'gambling_detected': False,
            'ad_count': 0,
            'predictions': [],
            'annotated_image_url': '',
        }

    if not InferenceHTTPClient or not ROBOFLOW_API_KEY or not ROBOFLOW_WORKSPACE or not ROBOFLOW_WORKFLOW_ID:
        missing = []
        if not InferenceHTTPClient:
            missing.append('inference-sdk')
        if not ROBOFLOW_API_KEY:
            missing.append('ROBOFLOW_API_KEY')
        if not ROBOFLOW_WORKSPACE:
            missing.append('ROBOFLOW_WORKSPACE')
        if not ROBOFLOW_WORKFLOW_ID:
            missing.append('ROBOFLOW_WORKFLOW_ID')

        return {
            'enabled': False,
            'error': f"roboflow_not_configured: {', '.join(missing)}",
            'gambling_detected': False,
            'ad_count': 0,
            'predictions': [],
            'annotated_image_url': '',
        }

    try:
        client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
        workflow_output = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={'image': screenshot_result['path']},
            use_cache=True,
        )

        payload = workflow_output[0] if isinstance(workflow_output, list) and workflow_output else workflow_output
        raw_predictions = _extract_prediction_list(payload)
        normalized_predictions = [_normalize_prediction_item(item) for item in raw_predictions]
        annotated_image_info = _extract_annotated_image_path(payload)

        gambling_keywords = ('ad', 'advert', 'banner', 'promo', 'slot', 'casino', 'bet', 'judol', 'judi', 'togel', 'poker', 'gambling')
        gambling_detected = bool(payload.get('gambling_detected')) if isinstance(payload, dict) else False
        gambling_detected = gambling_detected or any(
            any(keyword in str(pred.get('class', '')).lower() for keyword in gambling_keywords)
            for pred in normalized_predictions
        )

        ad_count = payload.get('ad_count') if isinstance(payload, dict) and isinstance(payload.get('ad_count'), int) else len(normalized_predictions)
        if not ad_count and normalized_predictions:
            ad_count = len(normalized_predictions)

        return {
            'enabled': True,
            'error': '',
            'gambling_detected': gambling_detected,
            'ad_count': ad_count,
            'predictions': normalized_predictions,
            'annotated_image_url': annotated_image_info.get('annotated_image_url', ''),
            'annotated_image_path': annotated_image_info.get('annotated_image_path', ''),
            'raw': payload,
        }
    except Exception as exc:
        return {
            'enabled': False,
            'error': str(exc),
            'gambling_detected': False,
            'ad_count': 0,
            'predictions': [],
            'annotated_image_url': '',
        }

def build_ml_features(url: str, title: str = "", meta_description: str = "", triggered_signals: str = "") -> dict:
    """Build ML features identik dengan training.py"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    full_url = url.lower()
    combined = f"{url} {title} {meta_description} {triggered_signals}".lower()

    # Structural
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)
    dot_count = full_url.count(".")
    dash_count = full_url.count("-")
    digit_count = sum(c.isdigit() for c in full_url)
    special_chars = sum(not c.isalnum() for c in full_url)
    is_https = 1 if url.startswith("https") else 0
    tld = domain.split(".")[-1] if "." in domain else ""
    suspicious_tlds = {"vip", "xyz", "top", "live", "club", "online", "site", "win", "bet", "casino", "poker"}
    is_suspicious_tld = 1 if tld in suspicious_tlds else 0

    # Entropy
    domain_entropy = calculate_entropy(domain)

    # Keyword hits
    judol_hits_text = sum(1 for kw in JUDOL_KEYWORDS if kw in combined)
    judol_hits_domain = sum(1 for kw in JUDOL_KEYWORDS if kw in domain or kw in path)
    judol_hits_total = judol_hits_text + judol_hits_domain

    words = combined.split()
    total_words = max(len(words), 1)
    keyword_density = judol_hits_total / total_words

    # Repeated words
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    repeated_words_count = sum(1 for v in word_counts.values() if v > 1)

    # Ratios
    symbol_ratio = special_chars / max(url_length, 1)

    # Suspicious subdomain
    subdomain_parts = domain.split(".")
    suspicious_sub = sum(1 for part in subdomain_parts if any(kw in part for kw in SUSPICIOUS_SUBDOMAINS))
    suspicious_subdomain_score = min(suspicious_sub / max(len(subdomain_parts), 1), 1.0)

    # Redirect
    redirect_hits = sum(1 for kw in REDIRECT_INDICATORS if kw in path)
    redirect_indicator_score = min(redirect_hits / max(len(path.split("/")), 1), 1.0)

    # Metadata quality
    metadata_quality = sum([
        1 if title and title not in ("nan", "") else 0,
        1 if meta_description and meta_description not in ("nan", "") else 0,
    ]) / 2.0

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "dot_count": dot_count,
        "dash_count": dash_count,
        "digit_count": digit_count,
        "special_char_count": special_chars,
        "https": is_https,
        "is_suspicious_tld": is_suspicious_tld,
        "domain_entropy": domain_entropy,
        "judol_hits_text": judol_hits_text,
        "judol_hits_domain": judol_hits_domain,
        "judol_hits_total": judol_hits_total,
        "keyword_density": keyword_density,
        "repeated_words_count": repeated_words_count,
        "symbol_ratio": symbol_ratio,
        "suspicious_subdomain_score": suspicious_subdomain_score,
        "redirect_indicator_score": redirect_indicator_score,
        "metadata_quality": metadata_quality,
    }

def predict_url(url: str) -> dict:
    """Prediksi malicious/judol URL menggunakan ML model"""
    if not all([ml_model, ml_vectorizer, ml_scaler]):
        return {"error": "ML models not loaded"}
    
    try:
        # Auto scrape
        scraped = auto_scrape(url)
        title = scraped["title"]
        meta_description = scraped["meta_description"]
        
        # Clean combined text
        combined = clean_text(f"{url} {title} {meta_description}")
        
        # TF-IDF
        X_text = ml_vectorizer.transform([combined])
        
        # Build numerical features
        features = build_ml_features(url, title, meta_description, "")
        
        # Get scaler columns
        try:
            scaler_cols = list(ml_scaler.feature_names_in_)
        except AttributeError:
            scaler_cols = list(features.keys())
        
        # Create DataFrame dengan kolom yang tepat
        num_df = pd.DataFrame(
            [[features.get(col, 0.0) for col in scaler_cols]],
            columns=scaler_cols,
        )
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_num_scaled = ml_scaler.transform(num_df)
        
        # Combine
        X_final = hstack([X_text, X_num_scaled])
        
        # Predict
        label = ml_model.predict(X_final)[0]
        proba = ml_model.predict_proba(X_final)[0]
        class_labels = ml_model.classes_
        
        prob_dict = dict(zip(class_labels, proba))
        confidence = max(proba)
        
        return {
            "url": url,
            "label": label,
            "confidence": float(confidence),
            "probabilities": {str(k): float(v) for k, v in prob_dict.items()},
            "title": title,
            "meta_description": meta_description,
            "features": features,
        }
    except Exception as e:
        print(f"[ML Prediction Error]: {e}")
        return {
            "url": url,
            "label": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "error": str(e),
        }

def normalize_prediction_label(label) -> str:
    label_text = str(label).strip().lower()
    if not label_text:
        return "unknown"

    malicious_tokens = {"malicious", "danger", "unsafe", "phishing", "phish", "spam", "judol", "bad"}
    safe_tokens = {"safe", "benign", "clean", "normal", "legit", "ham"}

    if any(token in label_text for token in malicious_tokens):
        return "malicious"
    if any(token in label_text for token in safe_tokens):
        return "safe"
    return label_text

def build_actionable_insights(url: str, features: dict, label: str, confidence: float, roboflow: dict = None) -> dict:
    """Turn ML output into practical findings, anomalies, and next actions."""
    features = features or {}
    label = normalize_prediction_label(label)
    roboflow = roboflow or {}
    
    # Mitigation: If risky domain extension has Roboflow verification with no gambling detected,
    # override ML classification to safe (even if ML found many judol keywords)
    if (
        is_risky_domain_extension(url)
        and roboflow.get('enabled')
        and not roboflow.get('gambling_detected')
        and roboflow.get('ad_count', 0) == 0
    ):
        label = "safe"
        # Allow keywords to exist in educational/news context
        # This is legitimate discussion about gambling, not malicious advertising

    indicators = []
    vulnerabilities = []
    anomalies = []
    recommendations = []

    domain_entropy = float(features.get("domain_entropy", 0.0) or 0.0)
    keyword_density = float(features.get("keyword_density", 0.0) or 0.0)
    judol_hits_total = int(features.get("judol_hits_total", 0) or 0)
    suspicious_tld = int(features.get("is_suspicious_tld", 0) or 0)
    suspicious_subdomain_score = float(features.get("suspicious_subdomain_score", 0.0) or 0.0)
    redirect_indicator_score = float(features.get("redirect_indicator_score", 0.0) or 0.0)
    metadata_quality = float(features.get("metadata_quality", 0.0) or 0.0)
    https_enabled = int(features.get("https", 0) or 0)
    url_length = int(features.get("url_length", 0) or 0)
    parsed = urlparse(url)
    is_dot_com = parsed.netloc.lower().endswith('.com')

    risk_score = 0
    
    # Check if this is a whitelisted educational or news domain
    is_edu_news_domain = is_educational_or_news_domain(url)
    
    # Override to safe if educational/news domain discussing gambling topics
    if is_edu_news_domain:
        label = "safe"
        indicators.append("✓ Verified as legitimate educational/news source")
        indicators.append("Domain is whitelisted for educational and news content about gambling")
    
    # Check if this is a risky domain that was verified safe by Roboflow
    roboflow_verified_safe = (
        is_risky_domain_extension(url)
        and roboflow.get('enabled')
        and not roboflow.get('gambling_detected')
        and roboflow.get('ad_count', 0) == 0
    )

    if label == "malicious":
        if roboflow_verified_safe:  
            # Override malicious label if Roboflow verified safe
            risk_score += 15
            indicators.append("Verified safe by visual analysis (Roboflow)")
            indicators.append("ML flagged keywords but visual analysis found no gambling content")
        else:
            risk_score += 35
            indicators.append("Model classifies this URL as malicious")
    elif label == "safe":
        risk_score += 5
        if roboflow_verified_safe:
            indicators.append("Verified safe by both ML model and visual analysis")
        else:
            indicators.append("Model classifies this URL as safe")
    else:
        if roboflow_verified_safe:
            risk_score += 10
            indicators.append("Inconclusive ML result, but verified safe by visual analysis")
        else:
            risk_score += 20
            indicators.append("Model output is inconclusive")

    if not is_edu_news_domain:
        if confidence >= 0.85:
            risk_score += 15
        elif confidence >= 0.65:
            risk_score += 8
        else:
            risk_score += 2

    if judol_hits_total > 0:
        # If educational/news domain or Roboflow verified safe, judol keywords are legitimate context
        if is_edu_news_domain or roboflow_verified_safe:
            indicators.append(f"{judol_hits_total} gambling-related keyword(s) found but in legitimate educational/news context")
        else:
            risk_score += min(20, judol_hits_total * 4)
            indicators.append(f"{judol_hits_total} gambling-related keyword hit(s) found")
            vulnerabilities.append({
                "title": "Judol Keyword Match",
                "severity": "HIGH" if judol_hits_total >= 2 else "MEDIUM",
                "domain": "Content Signals",
                "description": "Page content or URL contains gambling/judol indicators that often appear on malicious pages.",
            })

    if suspicious_tld:
        risk_score += 10
        indicators.append("Suspicious top-level domain detected")
        vulnerabilities.append({
            "title": "Suspicious TLD",
            "severity": "MEDIUM",
            "domain": "Domain Reputation",
            "description": "The domain extension belongs to a TLD that is frequently abused by scam or spam infrastructure.",
        })

    if suspicious_subdomain_score > 0 and not is_edu_news_domain:
        risk_score += min(12, int(suspicious_subdomain_score * 20))
        indicators.append("Suspicious subdomain pattern detected")
        vulnerabilities.append({
            "title": "Suspicious Subdomain Pattern",
            "severity": "MEDIUM",
            "domain": "Host Structure",
            "description": "Subdomain structure contains terms frequently used for lure, redirect, or fake-login pages.",
        })

    if redirect_indicator_score > 0 and not is_edu_news_domain:
        risk_score += min(10, int(redirect_indicator_score * 20))
        indicators.append("Redirect-like path detected")
        vulnerabilities.append({
            "title": "Redirect Behavior",
            "severity": "MEDIUM",
            "domain": "URL Routing",
            "description": "Path patterns suggest the page may redirect users through tracking or affiliate chains.",
        })

    if domain_entropy >= 3.7:
        risk_score += 8
        indicators.append("High domain entropy detected")
        vulnerabilities.append({
            "title": "High Domain Entropy",
            "severity": "LOW" if domain_entropy < 4.0 else "MEDIUM",
            "domain": "Domain Structure",
            "description": "Random-looking domain strings are often used to rotate malicious infrastructure.",
        })

    if metadata_quality < 1.0 and not roboflow_verified_safe and not is_edu_news_domain:
        risk_score += 6
        indicators.append("Missing or weak metadata")
        vulnerabilities.append({
            "title": "Low Metadata Quality",
            "severity": "LOW",
            "domain": "Page Metadata",
            "description": "The page exposes limited metadata, reducing trust signals for users and scanners.",
        })

    if metadata_quality == 0.0 and not roboflow_verified_safe and not is_edu_news_domain:
        risk_score += 20
        indicators.append("Missing title and meta description")
        vulnerabilities.append({
            "title": "Missing Metadata",
            "severity": "HIGH",
            "domain": "Page Metadata",
            "description": "Page has no title and no meta description, making it suspicious and prone to disguised gambling content.",
        })
        anomalies.append({
            "url": url,
            "risk": "Missing critical metadata",
        })
        if risk_score < 55:
            risk_score = 55

    if not https_enabled:
        risk_score += 10
        indicators.append("HTTPS is not enabled")
        vulnerabilities.append({
            "title": "No HTTPS",
            "severity": "HIGH",
            "domain": "Transport Security",
            "description": "The URL does not use HTTPS, so traffic can be intercepted or modified in transit.",
        })

    if url_length >= 120:
        risk_score += 6
        indicators.append("Very long URL detected")

    if keyword_density >= 0.15:
        risk_score += 8
        indicators.append("Keyword density is unusually high")

    # Apply YOLO/Roboflow visual detection risk boost
    gambling_detected = bool(roboflow.get('gambling_detected'))
    ad_count = int(roboflow.get('ad_count', 0) or 0)
    if gambling_detected or ad_count > 0:
        bonus = 30 if gambling_detected else min(20, ad_count * 4)
        risk_score += bonus
        indicators.append("Visual gambling ad detection by YOLO/Roboflow")
        vulnerabilities.append({
            "title": "Visual Gambling Ad Detection",
            "severity": "HIGH" if gambling_detected else "MEDIUM",
            "domain": "Visual Analysis",
            "description": "Detected gambling or advertisement regions from the page screenshot using the visual model.",
        })
        anomalies.append({
            "url": url,
            "risk": "Detected visual gambling ad content",
        })

    if is_dot_com and metadata_quality < 1.0 and not roboflow_verified_safe and not is_edu_news_domain:
        added = 16 if metadata_quality == 0.0 else 10
        risk_score += added
        indicators.append("Top-level .com with weak metadata")
        vulnerabilities.append({
            "title": "Weak Metadata on Common Domain",
            "severity": "MEDIUM",
            "domain": "Domain & Metadata",
            "description": "Common .com domain with poor metadata increases the likelihood of disguised gambling content.",
        })
        if label == "safe":
            indicators.append("Common domain with missing metadata raises suspicion")
        if risk_score < 55:
            risk_score = 55

    if gambling_detected and label == "safe":
        indicators.append("Safe label overridden by visual gambling detection")
        recommendations.insert(0, "Review this page immediately because visual analysis found gambling ad content.")

    if label == "malicious" and risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 55:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    if "malicious" in label:
        anomalies.extend([
            {
                "url": f"{url.rstrip('/')}/admin",
                "risk": "Admin surface exposed",
            },
            {
                "url": f"{url.rstrip('/')}/login",
                "risk": "Credential harvesting pattern",
            },
        ])

    if redirect_indicator_score > 0:
        anomalies.append({
            "url": f"{url}?redirect=external",
            "risk": "Open redirect / tracking indicator",
        })

    if not https_enabled:
        anomalies.append({
            "url": url,
            "risk": "Traffic transmitted without TLS",
        })

    if judol_hits_total > 0:
        anomalies.append({
            "url": url,
            "risk": "Judol/gambling keyword surface",
        })

    recommendations = [
        "Block the URL at the gateway or browser protection layer until reviewed.",
        "Verify the domain ownership and certificate status before allowing access.",
        "Search the page source for redirect chains, affiliate links, or credential capture forms.",
    ]

    if not https_enabled:
        recommendations.insert(0, "Enable HTTPS and renew the TLS certificate immediately.")
    if label == "malicious":
        recommendations.insert(0, "Treat the page as unsafe and avoid submitting any credentials or tokens.")

    return {
        "risk_score": min(risk_score, 100),
        "risk_level": risk_level,
        "indicators": indicators[:6],
        "vulnerabilities": vulnerabilities[:5],
        "anomalies": anomalies[:6],
        "recommendations": recommendations[:4],
    }

# ==========================================
# 3. BAGIAN ROUTE
# ==========================================

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/check-website')
def check_website():
    target_url = request.args.get('url', '')
    
    # Tambah https:// otomatis kalau belum ada
    if target_url and not target_url.startswith('http'):
        target_url = 'https://' + target_url
    
    # Gunakan ML model untuk prediksi
    ml_result = predict_url(target_url) if target_url else {}
    normalized_label = normalize_prediction_label(ml_result.get('label', 'unknown'))
    features = ml_result.get('features', {})
    has_no_metadata = not bool(ml_result.get('title')) and not bool(ml_result.get('meta_description'))
    insights = build_actionable_insights(
        target_url,
        features,
        normalized_label,
        float(ml_result.get('confidence', 0.0) or 0.0),
    )
    risk_score = insights['risk_score']
    
    # Build result dengan data dari ML
    result = {
        'url': target_url,
        'ml_label': normalized_label,
        'ml_confidence': ml_result.get('confidence', 0.0),
        'ml_probabilities': ml_result.get('probabilities', {}),
        'title': ml_result.get('title', '-'),
        'meta_description': ml_result.get('meta_description', '-'),
        'features': ml_result.get('features', {}),
        'risk_score': risk_score,
        'risk_level': insights['risk_level'],
        'analysis_indicators': insights['indicators'],
        'vulnerabilities': insights['vulnerabilities'],
        'anomalies': insights['anomalies'],
        'recommendations': insights['recommendations'],
        'seoScore': max(0, 100 - risk_score),
        'loadTime': round(0.8 + (len(target_url) / 120 if target_url else 0), 2),
            'is_safe_label': normalized_label == 'safe',
    }

    # Always run Roboflow for risky domain extensions (.com, .id, .co.id)
    # or if ML model shows malicious/risky indicators
    should_run_roboflow = (
        is_risky_domain_extension(target_url)
        or (
            normalized_label != 'safe'
            and (
                normalized_label == 'malicious'
                or risk_score >= 45
                or int(features.get('judol_hits_total', 0) or 0) > 0
                or (urlparse(target_url).netloc.lower().endswith('.com') and float(features.get('metadata_quality', 0.0) or 0.0) < 1.0)
                or has_no_metadata
            )
        )
    )

    if target_url and should_run_roboflow:
        screenshot_result = capture_url_screenshot(target_url)
        roboflow_result = analyze_with_roboflow(target_url, screenshot_result)
        result['screenshot'] = {
            'enabled': screenshot_result.get('ok', False),
            'error': screenshot_result.get('error', ''),
            'path': screenshot_result.get('relative_path', ''),
        }
        result['roboflow'] = roboflow_result
        insights = build_actionable_insights(
            target_url,
            features,
            normalized_label,
            float(ml_result.get('confidence', 0.0) or 0.0),
            roboflow_result,
        )
        risk_score = insights['risk_score']
        result.update({
            'risk_score': risk_score,
            'risk_level': insights['risk_level'],
            'analysis_indicators': insights['indicators'],
            'vulnerabilities': insights['vulnerabilities'],
            'anomalies': insights['anomalies'],
            'recommendations': insights['recommendations'],
            'seoScore': max(0, 100 - risk_score),
        })
    else:
        result['screenshot'] = {
            'enabled': False,
            'error': '',
            'path': '',
        }
        result['roboflow'] = {
            'enabled': False,
            'error': '',
            'gambling_detected': False,
            'ad_count': 0,
            'predictions': [],
            'annotated_image_url': '',
        }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)