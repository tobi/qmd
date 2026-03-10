# Linuxbrew/Node 25 Global Install Fix

## Problem
On linuxbrew (Homebrew on Linux) with Node 25+, the global npm bin wrapper for qmd hardcodes a wrong path:

Error:
\`\`\`
Error: Cannot find module \"/home/linuxbrew/.linuxbrew/dist/cli/qmd.js\"
\`\`\`

**Repro:**
1. \`npm install -g @tobilu/qmd@2.0.0\`
2. \`qmd --version\` → MODULE_NOT_FOUND

**Cause:** npm generates bin/qmd as \`exec node \"/home/linuxbrew/.linuxbrew/dist/cli/qmd.js\" \"\$@\"\`, but actual module is in \`lib/node_modules/@tobilu/qmd/dist/cli/qmd.js\` (no dist/ symlink created).

npx works fine.

## Workaround
\`\`\`bash
alias qmd=\"npx @tobilu/qmd@2.0.0\"
\`\`\`

## Proposed Fix
Add postinstall script to package.json:
\`\`\`json
{
  \"scripts\": {
    \"postinstall\": \"node -e \\\"const fs = require(\"fs\"); const prefix = require(\"npm-prefix\")(); const binPath = `${prefix}/bin/qmd`; const modPath = `${prefix}/lib/node_modules/@tobilu/qmd/dist/cli/qmd.js`; const content = \`#!/usr/bin/env node\\\\nrequire(\"${modPath}\");\\\`; fs.writeFileSync(binPath, content); fs.chmodSync(binPath, 0o755);\`\\\"
  }
}
\`\`\`

Logs: [qmd-install.log](https://pastebin.com/raw/example) (attach actual).

CC @tobi
