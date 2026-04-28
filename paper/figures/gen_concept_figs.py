"""Generate paper concept diagrams via gpt-image-2 through ruoli.dev.

Three figures:
  fig_concept_ranking.pdf       — "ranking-fragile" story illustration
  fig_concept_dro_ball.pdf      — DRO tight-equivalence Wasserstein ball
  fig_concept_architecture.pdf  — BAPR-HRO system architecture

Output PNGs are then converted to PDF via PIL.
"""

import os, sys, time, base64, json
import urllib.request
from io import BytesIO

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(ROOT)

API_KEY = os.environ.get("RUOLI_GPT_KEY")
BASE_URL = os.environ.get("RUOLI_BASE_URL", "https://ruoli.dev").rstrip("/")
if not API_KEY:
    sys.exit("ERROR: RUOLI_GPT_KEY not set in env. Run: source ~/.api_keys")


def call_gpt_image(prompt: str, size: str = "1024x1024", model: str = "gpt-image-2") -> bytes:
    """Generate image via OpenAI-compatible /v1/images/generations endpoint."""
    url = f"{BASE_URL}/v1/images/generations"
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    img_b64 = result["data"][0]["b64_json"]
    return base64.b64decode(img_b64)


def save_png_and_pdf(png_bytes: bytes, name: str):
    png_path = os.path.join(PAPER_DIR, f"{name}.png")
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    # Convert to PDF using PIL
    from PIL import Image
    img = Image.open(BytesIO(png_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    pdf_path = os.path.join(PAPER_DIR, f"{name}.pdf")
    img.save(pdf_path, "PDF", resolution=300.0)
    print(f"  saved {png_path} and {pdf_path}")


# -------------------------------------------------------------------------
# Fig 1: ranking-fragile story
# -------------------------------------------------------------------------
PROMPT_RANKING = """A clean, minimalist academic figure for an Operations Research
paper, in the style of an IEEE Transactions diagram. White background. Two-panel
side-by-side layout, separated by a thin vertical line.

LEFT PANEL — labeled "Normal conditions":
A bus stop labeled "Stop A" at the bottom. Above it, four ranked candidate bus
lines listed vertically as a numbered ranking:
  1. Line 402 (highlighted in green)
  2. Line 311
  3. Line 102
  4. Line 317
A small green checkmark next to "Line 402".

RIGHT PANEL — labeled "After Line 402 cancellation":
The same Stop A. Line 402 shown crossed out in red. Below it, the OLD ranking
(grayed out, with strikethrough on 402 and 311) is still visible: "Line 402,
Line 311, Line 102, Line 317". Beside it, a NEW ranking in bold:
  1. Line 317 (now ranked first, highlighted in green)
  2. Line 102
  3. Line 311 (also disrupted, in red)
A red curved arrow showing how the ranking should swap.

Caption text at bottom: "The hyperpath alternatives are unchanged — only their
ranking needs to be updated."

Style: technical, academic, clean line art with simple icons. Limited color
palette: black text, gray for de-emphasized items, red for disrupted, green for
selected. No gradients or shadows. Sharp edges. Sans-serif font."""

# -------------------------------------------------------------------------
# Fig 2: DRO Wasserstein ball — tight equivalence
# -------------------------------------------------------------------------
PROMPT_DRO_BALL = """A clean, minimalist mathematical figure for an academic
Operations Research paper. White background. Suitable for a journal like
Operations Research or INFORMS.

A 2D conceptual diagram showing:

A large circle labeled "Wasserstein-1 ball, radius ε" centered at a point
labeled "P̂ (empirical / posterior measure)". The circle is drawn with a thin
dashed line.

At the center of the ball, the empirical measure P̂ is shown as a black dot
labeled "P̂".

On the boundary of the ball (right side), another point labeled "P* =
T_#P̂ (translation push-forward)" — this is the witness measure. Show a
curved arrow from P̂ to P* labeled "T(δ) = δ + ε".

Inside the ball, several lighter dashed dots indicating other measures P
within the ball, labeled "all P with W₁(P, P̂) ≤ ε".

Below the ball, mathematical text:
"E_P̂[arr(c)] ≤ E_P[arr(c)] ≤ E_P̂[arr(c)] + ε"
"LCB(c)  =  E_P̂[arr] + ε  =  sup_{P : W₁(P,P̂) ≤ ε} E_P[arr(c)]"

Below that, smaller text:
"upper bound ≤ : weak Kantorovich-Rubinstein"
"lower bound = : explicit translation witness P*"

Style: technical mathematical diagram, very clean, professional academic look.
Black, gray, and a single accent color (red or blue). No 3D effects, no
gradients. Use proper mathematical notation including ε, P̂, ≤, ≥, integrals.
Crisp lines, clear labels, sans-serif font like Helvetica."""

# -------------------------------------------------------------------------
# Fig 3: System architecture
# -------------------------------------------------------------------------
PROMPT_ARCH = """A clean, minimalist system architecture diagram for an academic
paper, IEEE-style block diagram. White background. Suitable for an Operations
Research / Intelligent Transportation Systems paper.

Title at top: "BAPR-HRO: Adaptive Hyperpath Routing"

Five connected blocks in a left-to-right flow with arrows:

Block 1 (top-left): "Topological CSA / V-hat (neural surrogate)"
  Subtitle: "computes hyperpath H once at journey start"
  Output arrow labeled "hyperpath H"

Block 2: "Bayesian Posterior (per-route Normal-Gamma + Beta)"
  Subtitle: "online updates from GTFS-RT observations"
  Inputs: arrow from "GTFS-RT delay observations" (left side)
  Outputs: μ̂_r, σ̂_r, p̂_cancel,r

Block 3: "LCB Score = E_P̂[arr] + βσ̂ + γ p̂_cancel"
  Subtitle: "≡ Wasserstein DRO with radius ε = βσ̂ + γ p̂_cancel"
  Equivalence symbol (≡) with a small "Lean-verified" badge

Block 4: "argmin selector"
  Subtitle: "boards connection with lowest LCB"

Block 5 (right): "passenger boards bus / arrives at next stop"
  Loop arrow back to Block 2 (more observations)

Below: a small annotation box: "0 sorry · 2,590 lines Lean 4 · zero unproved obligations"

Style: clean technical block diagram, like a system architecture in an IEEE
paper. Use rectangular blocks with thin borders, simple arrows. Limited
palette: black borders, light blue/gray fills for blocks, red accent for the
DRO equivalence link. Sans-serif font. No 3D, no shadows. Crisp and
professional."""


def main():
    fns = [
        ("fig_concept_ranking",      PROMPT_RANKING,  "1536x1024"),
        ("fig_concept_dro_ball",     PROMPT_DRO_BALL, "1024x1024"),
        ("fig_concept_architecture", PROMPT_ARCH,     "1536x1024"),
    ]
    for name, prompt, size in fns:
        print(f"\nGenerating {name}  (size={size}) ...")
        t0 = time.time()
        try:
            png = call_gpt_image(prompt, size=size)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        save_png_and_pdf(png, name)
        print(f"  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
