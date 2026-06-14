"""
PhishGuard AI - Synthetic Dataset Generator
===========================================

Generates a realistic, labelled corpus of phishing and legitimate emails so
the models can be trained fully offline (no Kaggle account or internet
required). The templates are written to overlap deliberately - legitimate
mail also contains links, urgency and brand names - so the resulting model
has to learn real patterns instead of a trivial keyword rule.

Run directly to (re)create data/phishing_emails.csv:

    python data/generate_dataset.py --rows 6000
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import string

BRANDS = [
    ("PayPal", "paypal.com"), ("Amazon", "amazon.com"), ("Apple", "apple.com"),
    ("Microsoft", "microsoft.com"), ("Netflix", "netflix.com"),
    ("Google", "google.com"), ("DHL", "dhl.com"), ("FedEx", "fedex.com"),
    ("Chase", "chase.com"), ("Wells Fargo", "wellsfargo.com"),
    ("LinkedIn", "linkedin.com"), ("Dropbox", "dropbox.com"),
    ("Instagram", "instagram.com"), ("Coinbase", "coinbase.com"),
]

# Real, benign senders/services - legit mail is varied, not just one template.
LEGIT_SERVICES = [
    ("GitHub", "github.com", "notifications"), ("Slack", "slack.com", "team"),
    ("Uber", "uber.com", "receipts"), ("Spotify", "spotify.com", "billing"),
    ("Figma", "figma.com", "hello"), ("Notion", "notion.so", "team"),
    ("Zoom", "zoom.us", "no-reply"), ("Stripe", "stripe.com", "support"),
    ("Airbnb", "airbnb.com", "automated"), ("Calendly", "calendly.com", "notifications"),
    ("Grammarly", "grammarly.com", "info"), ("Atlassian", "atlassian.com", "jira"),
]

COMPANY_DOMAINS = ["acmecorp.com", "novatech.io", "brightlabs.co", "meridianhq.com",
                   "northwind.org", "bluepeak.net"]

FIRST_NAMES = ["alex", "sam", "jordan", "taylor", "morgan", "casey", "jamie",
               "priya", "arjun", "neha", "rahul", "diya", "ben", "sara", "leo"]

LEGIT_TLDS = [".com", ".org", ".net", ".io", ".co"]
BAD_TLDS = [".xyz", ".top", ".tk", ".zip", ".click", ".work", ".support"]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd"]

# Cheap, registrable two-word attacker domains on bad TLDs. These look
# "clean" (no digits, no token subdomain) - the ONLY tell is the bad TLD.
# Real phishing leans on exactly these (parcel-track.tk, rewards-claim.click).
CHEAP_WORDS_A = ["secure", "account", "rewards", "parcel", "billing", "verify",
                 "support", "service", "login", "payment", "delivery", "customer"]
CHEAP_WORDS_B = ["claim", "verify", "track", "portal", "center", "update",
                 "secure", "login", "alert", "desk", "team", "online"]


def cheap_domain(brand_l: str | None = None) -> str:
    """A short, clean-looking two-word domain on a bad TLD - no digits."""
    if brand_l and random.random() < 0.4:
        a = brand_l
    else:
        a = random.choice(CHEAP_WORDS_A)
    return f"{a}-{random.choice(CHEAP_WORDS_B)}{random.choice(BAD_TLDS)}"


def _rand_token(n: int) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def legit_url(domain: str) -> str:
    paths = ["", "account", "orders", "help", "settings", "billing/history",
             "support/cases", "notifications"]
    return f"https://www.{domain}/{random.choice(paths)}".rstrip("/")


def phishing_url(brand_domain: str) -> str:
    style = random.random()
    brand = brand_domain.split(".")[0]
    look = brand.replace("o", "0").replace("l", "1").replace("i", "1")
    if style < 0.22:  # raw IP
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{brand}/secure-login.php"
    if style < 0.40:  # url shortener
        return f"https://{random.choice(SHORTENERS)}/{_rand_token(6)}"
    if style < 0.62:  # CLEAN cheap two-word domain on a bad TLD (the hard case)
        return (f"http://{cheap_domain(brand)}/"
                f"{random.choice(['login','verify','secure','pay','account','update'])}")
    if style < 0.80:  # look-alike subdomain on a bad TLD (token subdomain)
        return (f"http://{brand}-secure.{_rand_token(5)}"
                f"{random.choice(BAD_TLDS)}/verify?id={_rand_token(10)}")
    if style < 0.90:  # brand as a subdomain of an attacker domain (@ trick + bad TLD)
        return (f"http://{brand}.account-{_rand_token(4)}.com@"
                f"{_rand_token(8)}{random.choice(BAD_TLDS)}/login")
    # look-alike (digit-substituted) domain on a bad TLD, long obfuscated path
    return (f"http://{look}-{random.choice(['billing','verify','secure'])}"
            f"{random.choice(BAD_TLDS)}/"
            f"{'/'.join(_rand_token(6) for _ in range(3))}?token={_rand_token(20)}")


# ---------------------------------------------------------------------------
# Email builders
# ---------------------------------------------------------------------------

def _legit_mailbox(base: str) -> str:
    """Real services frequently use digits in the local part (no-reply2,
    notifications365, bounce-7f3a). Injecting them into ~35% of legitimate
    senders stops `sender_has_digits` from becoming a phishing proxy."""
    r = random.random()
    if r < 0.20:
        return f"{base}{random.randint(2, 99)}"
    if r < 0.35:
        return f"{base}-{_rand_token(4)}"
    return base


def make_legit():
    kind = random.random()

    # --- 1) SaaS / service notification (links, brands, "account", "sign-in") ---
    if kind < 0.45:
        name, domain, mailbox = random.choice(LEGIT_SERVICES)
        sender = f"{_legit_mailbox(mailbox)}@{domain}"
        url = f"https://www.{domain}/{random.choice(['account','settings','app','inbox','billing'])}"
        n = random.choice(FIRST_NAMES).title()
        templates = [
            (f"New sign-in to your {name} account",
             f"Hi {n}, we noticed a new sign-in from Chrome on Windows. "
             f"If this was you, no action is needed. Review activity: {url}"),
            (f"Your {name} receipt",
             f"Thanks for your purchase. Your total was ${random.randint(5,90)}.{random.randint(10,99)}. "
             f"View the receipt any time: {url}"),
            (f"You have {random.randint(2,18)} unread notifications",
             f"Here is your weekly {name} digest. Open the app to catch up: {url}"),
            (f"Your {name} invoice is ready",
             f"Hi {n}, your subscription renewed successfully. No action needed. "
             f"View your invoice in account settings: {url}"),
            (f"{random.choice(FIRST_NAMES).title()} shared a file with you",
             f"You've been invited to view a document. Open it whenever you like: {url}"),
            # Legit mail that legitimately asks you to act / verify - mild urgency
            # WITHOUT any structural tell, so urgency alone can't mean phishing.
            (f"Please verify your email for {name}",
             f"Hi {n}, welcome aboard! Please confirm your email address to finish "
             f"setting up your account: {url} This link is valid for 24 hours."),
            (f"Action needed: accept your {name} invite",
             f"{random.choice(FIRST_NAMES).title()} invited you to a workspace. "
             f"Accept the invitation to get started: {url}"),
        ]
        subj, body = random.choice(templates)
        # ~30% of the time, trim to a terse one-liner so text_length overlaps phish.
        if random.random() < 0.30:
            body = body.split(":")[0].rstrip(" .") + f": {url}"
        return subj, sender, body, url, 0

    # --- 2) genuine person-to-person / work email (often short, sometimes long) ---
    if kind < 0.75:
        name = random.choice(FIRST_NAMES)
        domain = random.choice(COMPANY_DOMAINS)
        suffix = random.choice(["", "", str(random.randint(1, 99))])
        sender = f"{name}.{random.choice(FIRST_NAMES)}{suffix}@{domain}"
        url = (f"https://docs.{domain}/{_rand_token(6)}"
               if random.random() < 0.6 else "")
        templates = [
            ("Re: project timeline",
             "Thanks, that works for me. I'll circulate the revised plan tomorrow."
             + (f" Doc is here for reference: {url}" if url else "")),
            ("Notes from today's sync",
             "Hey, great call. I put the action items in the shared doc"
             + (f": {url}" if url else ".") + " Let me know if I missed anything."),
            ("Lunch next week?",
             "Are you free Thursday or Friday? Happy to grab something near the office."),
            ("Quick update on the deploy",
             "Deploy went out cleanly, all checks green."
             + (f" Dashboard: {url}" if url else "")),
            ("Re: contract review",
             "Looks good overall. I left a couple of comments in section 3 - mostly "
             "wording. Once those are in I'm happy to sign off"
             + (f". Latest copy: {url}" if url else ".")),
            ("thanks!",
             "Got it, appreciate the quick turnaround. Talk soon."),
        ]
        subj, body = random.choice(templates)
        return subj, sender, body, url, 0

    # --- 3) brand transactional mail (receipts, shipping, statements) ---
    brand, domain = random.choice(BRANDS)
    sender = f"{_legit_mailbox(random.choice(['no-reply','support','orders','service']))}@{domain}"
    n = random.choice(FIRST_NAMES).title()
    o = random.randint(100000, 999999)
    url = legit_url(domain) if random.random() < 0.9 else ""
    templates = [
        (f"Your {brand} order #{o} has shipped",
         f"Hi {n}, your order #{o} is on its way and will arrive in 2-3 days. "
         f"Track it from your account: {url}"),
        (f"Your monthly {brand} statement is ready",
         f"Hello {n}, your statement is now available. No action is needed. "
         f"Review it here when convenient: {url}"),
        ("Your password was changed",
         f"Hi {n}, this confirms your password was changed. If this was you, "
         f"no action is needed. Manage security settings: {url}"),
        (f"Your {brand} payment receipt",
         f"We received your payment of ${random.randint(8,120)}.{random.randint(10,99)}. "
         f"Thanks for being a customer. Receipt: {url}"),
    ]
    subj, body = random.choice(templates)
    return subj, sender, body, url, 0


def _phish_sender(brand_l: str, look: str, clean: bool) -> str:
    """Build a phishing sender. When `clean` is True the sender looks
    plausible (no digits, no bad TLD) so the URL must carry the signal -
    this decorrelates the sender tell from the rest."""
    if clean:
        # Plausible-looking but not the real brand domain, no digits.
        return (f"{random.choice(['support','service','no-reply','team','help'])}"
                f"@{random.choice([f'{brand_l}-care', 'account-services', 'mail-secure', f'{brand_l}mail'])}.com")
    s = random.random()
    if s < 0.30:
        return (f"{random.choice(['security','account','service','alert','support'])}"
                f"{random.randint(1,99)}@{look}-{random.choice(['secure','verify','billing','team'])}"
                f"{random.choice(BAD_TLDS)}")
    if s < 0.52:
        # CLEAN dotted-name local part on a cheap bad-TLD domain, NO digits.
        # e.g. dhl.delivery@parcel-track.tk - only the TLD betrays it.
        local = random.choice(
            [f"{brand_l}.{random.choice(['support','service','account','delivery','billing'])}",
             f"{random.choice(['hr','it','admin','no-reply','team'])}.{random.choice(CHEAP_WORDS_B)}",
             random.choice(['support', 'service', 'no-reply', 'security'])])
        return f"{local}@{cheap_domain(brand_l)}"
    if s < 0.75:
        return f"{brand_l}.{random.choice(['support','security','service'])}{random.randint(1,999)}@gmail.com"
    if s < 0.88:
        return f"{random.choice(['hr','payroll','it','admin'])}@{_rand_token(5)}-portal{random.choice(BAD_TLDS)}"
    return f"{_rand_token(6)}@{look}{_rand_token(3)}{random.choice(BAD_TLDS)}"


def make_phishing():
    brand, domain = random.choice(BRANDS)
    brand_l = brand.lower().replace(" ", "")
    # Homoglyph / digit substitution for look-alike domains.
    look = brand_l.replace("o", "0").replace("l", "1").replace("i", "1")

    # Independent tell injection -------------------------------------------
    # The URL ALWAYS carries a structural tell (it is the ground-truth signal).
    # The sender tell and the text loudness are decided INDEPENDENTLY so the
    # model cannot collapse them into a single proxy. ~30% of phish have a
    # clean-looking sender, forcing reliance on the URL; ~25% use minimal,
    # polite text so the structural tells alone must carry the label.
    clean_sender = random.random() < 0.30
    sender = _phish_sender(brand_l, look, clean_sender)
    url = phishing_url(domain)

    style = random.random()
    if style < 0.35:        # LOUD social-engineering text
        templates = [
            (f"URGENT: Your {brand} account has been suspended",
             "Dear Customer, we detected UNUSUAL ACTIVITY on your account. It has "
             "been SUSPENDED. You must VERIFY your identity within 24 hours or it "
             f"will be permanently closed!!! Verify now: {url} Enter your password "
             "and billing details to restore access."),
            ("Action Required: Confirm your payment information",
             "Your recent payment could not be processed. Update your credit card "
             f"and account number IMMEDIATELY to avoid service interruption: {url} "
             "Failure to act will result in account termination!!!"),
            ("Congratulations! You have won a $1000 gift card",
             "WINNER! You have been selected to receive a $1000 reward. Claim your "
             f"prize now before it expires: {url} Provide your details to receive "
             "your cash reward immediately."),
        ]
    elif style < 0.70:      # CALM credential-harvesting text
        templates = [
            (f"Confirm your {brand} identity",
             f"Hello, your account needs to be verified. Please click here to "
             f"confirm your identity and enter your credentials: {url} "
             "Thank you for your cooperation."),
            ("Update your direct deposit",
             f"Please confirm your bank account and routing number to receive this "
             f"month's salary on time. Use the secure portal here: {url}"),
            (f"Your {brand} package is on hold",
             f"Your parcel is on hold due to an unpaid customs fee. Pay now and "
             f"update your billing information to release it: {url}"),
            ("Verify your wallet",
             f"We noticed unusual activity on your wallet. Confirm your seed phrase "
             f"and login credentials to secure your funds: {url}"),
        ]
    else:                   # MINIMAL / neutral text - only the structure betrays it
        templates = [
            (f"Your {brand} account",
             f"Please review your recent activity here: {url}"),
            ("Document shared with you",
             f"A document has been shared with you. Open it here: {url}"),
            (f"{brand} notification",
             f"There is an update on your account. View it here: {url}"),
            ("Sign-in confirmation",
             f"Confirm your recent sign-in to continue: {url}"),
            ("Re: your request",
             f"Following up - please complete the verification here: {url}"),
        ]
    subj, body = random.choice(templates)
    return subj, sender, body, url, 1


def generate(rows: int, out_path: str, seed: int = 42):
    random.seed(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "sender", "content", "urls", "label"])
        for _ in range(rows):
            subj, sender, body, url, label = (
                make_phishing() if random.random() < 0.5 else make_legit()
            )
            # ~2% label noise reflects real-world annotation error
            if random.random() < 0.02:
                label = 1 - label
            writer.writerow([subj, sender, body, url, label])
    print(f"Wrote {rows} rows -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic email dataset")
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "phishing_emails.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.rows, args.out, args.seed)
