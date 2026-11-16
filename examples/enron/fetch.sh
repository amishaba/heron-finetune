#!/usr/bin/env bash
# Downloads the Enron email corpus (CMU release of 2015-05-07, ~420 MB; public, released by FERC)
# and unpacks it to data/enron/maildir/<employee>/<folder>/<n>.
set -euo pipefail
cd "$(dirname "$0")/../.."
dest=data/enron
mkdir -p "$dest"
archive="$dest/enron_mail_20150507.tar.gz"
[ -f "$archive" ] || curl -fSL https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -o "$archive"
[ -d "$dest/maildir" ] || tar -xzf "$archive" -C "$dest"
echo "$dest/maildir: $(ls "$dest/maildir" | wc -l | tr -d ' ') mailboxes"
