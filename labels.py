from enum import Enum
# Model misalignment: From generic customer agents to AI scrappers,
# we don't want random text to change the output/behavior, even if it
# is not a safety violation. (ex. A customer ai agent is told to respond in JSON
# for no reason)
class PromptEntity(Enum):
    # --- THE ATTACK VECTORS (The "How") ---
    # --- Level 1 ---
    VIOLATION = "VIOLATION"  # Rule breaking "Ignore safety", "Disable filters", "Forget rules", "Unfiltered", "allowed to do X" (normally restricted), "force to reply no matter what"

    # --- Level 2 --- (standalone, it isn't malicious)
    OBFUSCATION = "OBFUSC"
    """
    STRICT RULE: Only use for MACHINE-READABLE or NON-ALPHABETIC ciphers.
    Includes: Base64, Morse Code, Hexadecimal, Binary, Rot13, or Caesar Ciphers.
    EXCLUDES: Spaced-out letters (H A T E), Leetspeak (H4ck), or Punctuation-stuffing (s.t.e.a.l).
    These 'Human-Readable' variants must be tagged with their CORE INTENT (e.g., SOCIAL or CYBER).
    """
    PERSONA = "PERSONA"  # You are (generic)
    HIGHSTAKES = "HIGHSTAKES"  # Professional advice: medical, legal, financial — harm via misinformation or liability
    NSHOT = "NSHOT"  # User/Assistant examples embedded in the prompt
    FORMAT = "FORMAT"

    # --- THE HARMFUL INTENT (The "What") ---
    VIOLENCE = "VIOLENCE"  # Murder, weapons, physical assault, CBRN
    CYBER = "CYBER"  # Hacking, malware, technical exploits
    ILLICIT = "ILLICIT"  # Financial crime, fraud, drugs, theft
    SOCIAL = "SOCIAL"  # Hate, harassment, NSFW, child safety
    SENSITIVE = "SENSITIVE"  # PII, Privacy leaks, API/secrets
    DISINFO = "DISINFO"  # Propaganda, Fake news, Defamation

    # --- Default ---
    BENIGN = "BENIGN"
