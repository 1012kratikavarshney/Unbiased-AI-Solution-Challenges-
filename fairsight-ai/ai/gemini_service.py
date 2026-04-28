import os
import json
import urllib.request
import urllib.error


class GeminiService:

    def __init__(self, api_key=None):
        self.api_key = None
        self.available = False
        key = api_key or os.getenv('GEMINI_API_KEY', '')
        if key:
            self._init_model(key)

    def _init_model(self, key):
        try:
            # Test the key with a direct REST call to v1 (not v1beta)
            url = (
                f"https://generativelanguage.googleapis.com/v1/models"
                f"?key={key}"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            # Pick best available model from the account
            available_models = [m['name'] for m in data.get('models', [])]
            preferred = [
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
                'models/gemini-pro',
            ]
            self.model_name = None
            for p in preferred:
                if p in available_models:
                    self.model_name = p
                    break
            if not self.model_name and available_models:
                self.model_name = available_models[0]
            if not self.model_name:
                return False, "No models available for this API key."
            self.api_key = key
            self.available = True
            return True, f"Gemini API connected. Using {self.model_name}"
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            self.available = False
            return False, f"Failed to connect: {e.code} {body}"
        except Exception as e:
            self.available = False
            return False, f"Failed to connect: {str(e)}"

    def set_api_key(self, api_key):
        if not api_key or not api_key.strip():
            self.api_key = None
            self.available = False
            return False, "API key cannot be empty."
        return self._init_model(api_key.strip())

    def _generate(self, prompt):
        """Direct REST call to Gemini v1 — no SDK, no v1beta."""
        url = (
            f"https://generativelanguage.googleapis.com/v1/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def explain_bias(self, results, feature, domain):
        if not self.available:
            return (
                "Please add your Gemini API key "
                "in the sidebar to get AI explanations."
            )
        score = results.get('fairness_score', 0)
        dpd = results.get('demographic_parity_difference', 0)
        groups = results.get('group_accuracies', {})
        prompt = f"""
You are FairSight AI — an expert in AI fairness.
Explain this bias analysis to a non-technical {domain} manager.

Analysis Data:
- Sensitive Feature Analyzed: {feature}
- Fairness Score: {score}/100
- Demographic Parity Difference: {dpd}
- Group Accuracies: {groups}
- Domain: {domain}

Instructions:
1. Start with ONE bold sentence summarizing the problem
2. Explain WHO is being disadvantaged and BY HOW MUCH
3. Give ONE real-world example of the harm this causes
4. State HOW SERIOUS this is (critical/moderate/minor)
5. Give ONE immediate action they should take

Rules:
- No technical jargon at all
- Maximum 150 words
- Be direct and honest
- Use simple everyday language
"""
        try:
            return self._generate(prompt)
        except Exception as e:
            return f"AI explanation unavailable. Error: {str(e)}"

    def chat(self, question, bias_context):
        if not self.available:
            return "Please add Gemini API key to use chat."
        prompt = f"""
You are FairSight AI Assistant — AI fairness expert.
Current bias analysis context: {bias_context}
User question: {question}

Answer in simple, practical terms.
Maximum 100 words. No jargon.
If they ask about legality, mention relevant laws simply.
If they ask about fixing, give specific steps.
"""
        try:
            return self._generate(prompt)
        except Exception as e:
            return f"Please retry. API temporarily unavailable. Error: {str(e)}"

    def generate_report(self, org_name, domain, results, fix):
        if not self.available:
            return self._fallback_report(org_name, domain, results)
        score = results.get('fairness_score', 0)
        dpd = results.get('demographic_parity_difference', 0)
        groups = results.get('group_accuracies', {})
        prompt = f"""
Write a professional AI Bias Audit Report.

Organization: {org_name}
Domain: {domain}
Analysis Date: Today
Fairness Score: {score}/100
Demographic Parity Difference: {dpd}
Group Accuracies: {groups}
Debiasing Applied: {fix}

Write these exact sections with these headings:
EXECUTIVE SUMMARY
BIAS FINDINGS
REAL-WORLD IMPACT
COMPLIANCE STATUS
RECOMMENDATIONS
CONCLUSION

Tone: Professional and formal.
Length: 300-350 words total.
Include specific numbers from the analysis.
Make compliance section mention relevant data
protection and anti-discrimination laws.
"""
        try:
            return self._generate(prompt)
        except Exception:
            return self._fallback_report(org_name, domain, results)

    def _fallback_report(self, org, domain, results):
        score = results.get('fairness_score', 'N/A')
        dpd = results.get('demographic_parity_difference', 'N/A')
        return f"""AI BIAS AUDIT REPORT
Organization: {org}
Domain: {domain}

FINDINGS:
Fairness Score: {score}/100
Demographic Parity Difference: {dpd}

RECOMMENDATION:
{"Immediate debiasing required." if isinstance(score, (int, float)) and score < 65 else "Continue monitoring."}

Note: Add Gemini API key for detailed AI-generated report."""