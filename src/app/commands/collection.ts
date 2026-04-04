export async function handleCollectionCommand(
  args: string[],
  values: Record<string, unknown>,
  deps: {
    getPwd: () => string;
    getRealPath: (path: string) => string;
    resolve: (...paths: string[]) => string;
    defaultGlob: string;
    collectionList: () => void;
    collectionAdd: (pwd: string, globPattern: string, name?: string) => Promise<void>;
    collectionRemove: (name: string) => void;
    collectionRename: (oldName: string, newName: string) => void;
  }
): Promise<void> {
  const subcommand = args[0];
  switch (subcommand) {
    case "list":
      deps.collectionList();
      return;

    case "add": {
      const pwd = args[1] || deps.getPwd();
      const resolvedPwd = pwd === "." ? deps.getPwd() : deps.getRealPath(deps.resolve(pwd));
      const globPattern = (values.mask as string) || deps.defaultGlob;
      const name = values.name as string | undefined;
      await deps.collectionAdd(resolvedPwd, globPattern, name);
      return;
    }

    case "remove":
    case "rm":
      if (!args[1]) {
        console.error("Usage: qmd collection remove <name>");
        console.error("  Use 'qmd collection list' to see available collections");
        process.exit(1);
      }
      deps.collectionRemove(args[1]);
      return;

    case "rename":
    case "mv":
      if (!args[1] || !args[2]) {
        console.error("Usage: qmd collection rename <old-name> <new-name>");
        console.error("  Use 'qmd collection list' to see available collections");
        process.exit(1);
      }
      deps.collectionRename(args[1], args[2]);
      return;

    default:
      console.error(`Unknown subcommand: ${subcommand}`);
      console.error("Available: list, add, remove, rename");
      process.exit(1);
  }
}
