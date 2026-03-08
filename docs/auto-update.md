# Keeping Your Index Fresh

As you add or edit notes, run:

```sh
qmd update   # re-scan collections for new/changed files
qmd embed    # generate embeddings for any new content
```

To automate this, schedule both commands to run periodically.

## macOS (launchd)

```xml
<!-- ~/Library/LaunchAgents/com.yourname.qmd-sync.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.yourname.qmd-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>qmd update &amp;&amp; qmd embed</string>
    </array>
    <key>StartInterval</key>
    <integer>14400</integer>
    <key>StandardOutPath</key>
    <string>/tmp/qmd-sync.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/qmd-sync.log</string>
</dict>
</plist>
```

```sh
launchctl load ~/Library/LaunchAgents/com.yourname.qmd-sync.plist
```

## Linux (cron)

```sh
crontab -e
# add:
0 */4 * * * qmd update && qmd embed >> /tmp/qmd-sync.log 2>&1
```
