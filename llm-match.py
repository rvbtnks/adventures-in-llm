#!/usr/bin/env python3
"""
Wrapper for fuzzy matcher that uses an LLM via Ollama to make match decisions
"""

import pexpect
import sys
import re
import requests
import argparse
import time
from typing import Optional, List, Dict

ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')
SCRIPT_TIMEOUT = 300
REQUEST_TIMEOUT = 30

PATTERNS = [
    r'Processing: (.+)',                    # 0: new folder
    r'Best match: (.+) \((\d+\.?\d*)%\)',   # 1: high confidence match
    r'Alt: (.+) \((\d+\.?\d*)%\)',          # 2: alternative match
    r'Lower confidence matches:',            # 3: low confidence header
    r'  (.+) \((\d+\.?\d*)%\)',             # 4: low confidence match
    r'Use this match\? \(y/n\): ',          # 5: decision prompt
    pexpect.EOF,                             # 6
    pexpect.TIMEOUT                          # 7
]


class OllamaMatcherWrapper:
    def __init__(self, ollama_host: str, model: str, debug: bool = False):
        self.ollama_url = f"http://{ollama_host}/api/generate"
        self.model = model
        self.debug = debug

    def debug_print(self, msg: str, data=None):
        if not self.debug:
            return
        print(f"\n[DEBUG] {msg}", file=sys.stderr)
        if data:
            print(f"[DEBUG DATA] {data}", file=sys.stderr)

    def test_connection(self) -> bool:
        print(f"\n{'='*60}")
        print(f"Testing connection to Ollama...")
        print(f"Host: {self.ollama_url}")
        print(f"Model: {self.model}")
        print(f"{'='*60}\n")

        try:
            print("Sending test query...")
            start = time.time()
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": "What is 2+2? Reply with only the number.", "stream": False, "temperature": 0.1},
                timeout=REQUEST_TIMEOUT
            )
            elapsed = time.time() - start

            if response.status_code != 200:
                print(f"❌ Error: HTTP {response.status_code}\n{response.text}")
                return False

            answer = response.json().get('response', '').strip()
            print(f"✅ Connection successful! ({elapsed:.2f}s)\nResponse: {answer}\n{'='*60}\n")
            return True

        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed: Cannot reach {self.ollama_url}")
        except requests.exceptions.Timeout:
            print(f"❌ Timeout: No response within {REQUEST_TIMEOUT}s")
        except Exception as e:
            print(f"❌ Error: {e}")
        return False

    def _parse_match(self, match_obj, match_type: str) -> Dict:
        title = ANSI_PATTERN.sub('', match_obj.group(1).strip())
        score = float(match_obj.group(2))
        return {'title': title, 'score': score, 'type': match_type}

    def ask_llm(self, folder_name: str, matches: List[Dict]) -> Optional[int]:
        match_list = '\n'.join(f"{i+1}. {m['title']} ({m['score']}%)" for i, m in enumerate(matches))
        prompt = f"""You are helping match folder names to database entries.
Given a folder name and possible matches with confidence scores, pick the best match.

Folder name: {folder_name}

Possible matches:
{match_list}

Reply with ONLY the number of the best match (1, 2, 3, etc.) or "SKIP" if none are good matches.
Consider both the match score and how well the titles actually match semantically."""

        self.debug_print("Sending to LLM:", prompt)

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "system": "You are a file matching assistant. Reply with only the match number or SKIP."
                },
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code != 200:
                self.debug_print(f"Ollama error: {response.status_code}")
                return None

            answer = response.json().get('response', '').strip()
            self.debug_print(f"LLM response: {answer}")

            if answer.upper() == "SKIP":
                return -1

            if match := re.search(r'\d+', answer):
                choice = int(match.group())
                if 1 <= choice <= len(matches):
                    return choice - 1

            self.debug_print(f"Failed to parse response: {answer}")

        except (requests.RequestException, ValueError) as e:
            self.debug_print(f"Error calling LLM: {e}")

        return None

    def _handle_decision(self, child, folder: str, matches: List[Dict]):
        self.debug_print(f"Need decision for {folder} with {len(matches)} matches")
        choice_idx = self.ask_llm(folder, matches)

        if choice_idx is None:
            print(f"→ LLM failed to respond for {folder}")
            child.sendline('n')
        elif choice_idx == -1:
            print(f"→ LLM skipped (no good matches)")
            child.sendline('n')
        else:
            chosen = matches[choice_idx]
            print(f"→ LLM selected: {chosen['title']} ({chosen['score']}%)")
            child.sendline('y')
            # Skip remaining matches
            for _ in range(len(matches) - choice_idx - 1):
                child.expect(PATTERNS[5])
                child.sendline('n')

    def run(self, script_path: str, args: List[str], dry_run: bool = False, verbose: bool = False):
        cmd = f"python3 {script_path} {' '.join(args)}"

        if dry_run:
            print(f"[DRY RUN] Would execute: {cmd}")
            return

        print(f"Running: {cmd}")
        child = pexpect.spawn(cmd, encoding='utf-8', timeout=SCRIPT_TIMEOUT)

        if verbose:
            child.logfile_read = sys.stdout

        folder = None
        matches = []
        match_type = None

        try:
            while True:
                idx = child.expect(PATTERNS)

                if idx == 0:  # new folder
                    folder = ANSI_PATTERN.sub('', child.match.group(1).strip())
                    self.debug_print(f"Processing folder: {folder}")
                    matches = []
                    match_type = None

                elif idx == 1:  # best match
                    matches = [self._parse_match(child.match, 'best')]
                    match_type = 'high'
                    self.debug_print(f"Found best match: {matches[0]}")

                elif idx == 2:  # alt match
                    matches.append(self._parse_match(child.match, 'alt'))
                    self.debug_print(f"Found alt match: {matches[-1]}")

                elif idx == 3:  # low confidence header
                    match_type = 'low'
                    matches = []

                elif idx == 4 and match_type == 'low':  # low match
                    matches.append(self._parse_match(child.match, 'low'))
                    self.debug_print(f"Found low match: {matches[-1]}")

                elif idx == 5:  # decision prompt
                    if matches and folder:
                        self._handle_decision(child, folder, matches)
                    matches = []

                elif idx in (6, 7):  # EOF or timeout
                    if idx == 7:
                        self.debug_print("Timeout waiting for output")
                    break

        except Exception as e:
            print(f"\n[ERROR] {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

        child.close()
        return child.exitstatus


def main():
    parser = argparse.ArgumentParser(description='Fuzzy matcher wrapper with LLM integration')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--verbose', action='store_true', help='Show all script output')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed')
    parser.add_argument('--ollama-host', default='localhost:11434', help='Ollama host:port')
    parser.add_argument('--model', default='llama3:8b', help='Model name')
    parser.add_argument('--test-connection', action='store_true', help='Test Ollama connection and exit')

    args, remaining = parser.parse_known_args()

    wrapper = OllamaMatcherWrapper(
        ollama_host=args.ollama_host,
        model=args.model,
        debug=args.debug
    )

    if args.test_connection:
        sys.exit(0 if wrapper.test_connection() else 1)

    script_path = remaining[0] if remaining and not remaining[0].startswith('-') else 'im.db.py'
    script_args = remaining[1:] if remaining and not remaining[0].startswith('-') else remaining

    exit_code = wrapper.run(script_path, script_args, dry_run=args.dry_run, verbose=args.verbose)
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
