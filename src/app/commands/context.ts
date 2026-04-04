export async function handleContextCommand(
  args: string[],
  deps: {
    contextAdd: (pathArg: string | undefined, contextText: string) => Promise<void>;
    contextList: () => void;
    contextCheck: () => void;
    contextRemove: (pathArg: string) => void;
  }
): Promise<void> {
  const subcommand = args[0];
  if (!subcommand) {
    console.error("Usage: qmd context <add|list|check|rm>");
    console.error("");
    console.error("Commands:");
    console.error("  qmd context add [path] \"text\"  - Add context (defaults to current dir)");
    console.error("  qmd context add / \"text\"       - Add global context to all collections");
    console.error("  qmd context list                - List all contexts");
    console.error("  qmd context check               - Check for missing contexts");
    console.error("  qmd context rm <path>           - Remove context");
    process.exit(1);
  }

  switch (subcommand) {
    case "add": {
      if (args.length < 2) {
        console.error("Usage: qmd context add [path] \"text\"");
        console.error("");
        console.error("Examples:");
        console.error("  qmd context add \"Context for current directory\"");
        console.error("  qmd context add . \"Context for current directory\"");
        console.error("  qmd context add /subfolder \"Context for subfolder\"");
        console.error("  qmd context add / \"Global context for all collections\"");
        console.error("");
        console.error("  Using virtual paths:");
        console.error("  qmd context add qmd://journals/ \"Context for entire journals collection\"");
        console.error("  qmd context add qmd://journals/2024 \"Context for 2024 journals\"");
        process.exit(1);
      }

      let pathArg: string | undefined;
      let contextText: string;

      const firstArg = args[1] || "";
      const secondArg = args[2];
      if (secondArg) {
        pathArg = firstArg;
        contextText = args.slice(2).join(" ");
      } else {
        pathArg = undefined;
        contextText = firstArg;
      }

      await deps.contextAdd(pathArg, contextText);
      return;
    }

    case "list":
      deps.contextList();
      return;

    case "check":
      deps.contextCheck();
      return;

    case "rm":
    case "remove":
      if (args.length < 2 || !args[1]) {
        console.error("Usage: qmd context rm <path>");
        console.error("Examples:");
        console.error("  qmd context rm /");
        console.error("  qmd context rm qmd://journals/2024");
        process.exit(1);
      }
      deps.contextRemove(args[1]);
      return;

    default:
      console.error(`Unknown subcommand: ${subcommand}`);
      console.error("Available: add, list, check, rm");
      process.exit(1);
  }
}
